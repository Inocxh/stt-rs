use std::cmp::{max, min};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::{BufRead, stdout, Write};
use std::num::ParseIntError;
use std::path::PathBuf;
use std::process::exit;
use std::time::{Duration, Instant};

use clap::Parser;
use itertools::Itertools;
use petgraph;
use petgraph::data::FromElements;
use petgraph::graph;
use petgraph::graph::UnGraph;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use stt::common::UsizeMaxMonoidWeightWithMaxEdge;
use stt::DynamicForest;
use stt::generate::generate_edge;
use stt::link_cut::MonoidLinkCutTree;
use stt::mst::compute_mst;
use stt::onecut::SimpleDynamicTree;
use stt::pg::PetgraphDynamicForest;
use stt::twocut::mtrtt::*;
use stt::twocut::splaytt::*;

use stt_benchmarks::bench_util::{ImplDesc, ImplName, PrintType};
use stt_benchmarks::bench_util::PrintType::{Json, Print};
use stt_benchmarks::do_for_impl_monoid;

type MSTWeight = UsizeMaxMonoidWeightWithMaxEdge;
type Edge = (usize, usize);
type EdgeWithWeight = (usize, usize, usize);


// Helper function
fn normalize_edge_vector( vec : &Vec<(usize, usize)> ) -> Vec<(usize, usize)> {
	return vec.iter().map( |(u,v)| ( min(*u,*v), max(*u,*v) ) ).sorted().collect();
}

fn generate_complete_edges_with_weights( num_vertices : usize, rng : &mut impl Rng )
		-> Vec<EdgeWithWeight>
{
	let mut edges : Vec<_> = (0..num_vertices).combinations( 2 ).collect();
	edges.shuffle( rng );
	let mut weights : Vec<_> = (0..edges.len()).collect();
	weights.shuffle( rng );
	(0..edges.len()).map( |i| {
		( edges[i][0], edges[i][1], weights[i] )
	} ).collect()
}

fn generate_sparse_edges_with_weights( num_vertices : usize, num_edges : usize, rng : &mut impl Rng )
		-> Vec<EdgeWithWeight>
{
	assert!( num_edges <= num_vertices * (num_vertices-1 ) / 2,
			"Cannot generate graph with {num_vertices} vertices and {num_edges} edges" );
	
	if num_edges >= num_vertices.pow(2) / 4 {
		return generate_complete_edges_with_weights( num_vertices, rng )[0..num_edges].to_vec()
	}
	
	let mut weights : Vec<_> = (0..num_edges).collect();
	weights.shuffle( rng );
	
	let mut seen_edges : HashSet<(usize, usize)> = HashSet::new();
	let mut result : Vec<EdgeWithWeight> = vec![];
	while result.len() < num_edges {
		let e = generate_edge( num_vertices, rng );
		if !seen_edges.contains( &e ) {
			seen_edges.insert( e );
			seen_edges.insert( (e.1, e.0) );
			let weight = weights[result.len()];
			result.push( ( e.0, e.1, weight ) );
		}
	}
	result
}

fn read_mst( path : &PathBuf ) -> io::Result<(usize, Vec<EdgeWithWeight>)> {
	let file = File::open( path )?;
	let mut num_vertices = 0;
	let mut edges : Vec<EdgeWithWeight> = vec![];
	for line in io::BufReader::new( file ).lines() {
		let line = line?;
		let parts : Vec<_> = line.split( " " ).collect();
		if parts[0] == "mst" {
			// "mst <num_vertices> <num_edges>"
			if parts.len() == 3 {
				if let Ok( n ) = parts[1].parse() {
					// Ignore number of edges
					num_vertices = n;
					continue;
				}
			}
			return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
		}
		else if parts[0] == "e" {
			// "e <from> <to> <weight>"

			fn parse_edge( edge_parts : &Vec<&str>) -> Result<EdgeWithWeight, ParseIntError> {
				let u = edge_parts[1].parse()?;
				let v = edge_parts[2].parse()?;
				let w = edge_parts[3].parse()?;
				return Ok( (u,v,w) )
			}
			
			if parts.len() == 4 {
				if let Ok( e ) = parse_edge( &parts ) {
					edges.push( e );
					continue
				}
			}
			return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
		}
		else if parts[0] == "c" {}
		else {
			return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
		}
	}
	Ok( (num_vertices, edges) )
}


/// Helper struct to store configuration, aggregate results, etc.
struct Helper {
	num_vertices : usize,
	input_edges : Vec<EdgeWithWeight>,
	input_edge_weights : HashMap<Edge, usize>,
	verify : bool,
	print : PrintType,
	allow_multi_edges : bool,
	verification_total_weight : Option<usize>,
	verification_edges: Option<Vec<(usize, usize)>>,
	iterations: usize
}

impl Helper {
	fn new( num_vertices : usize, input_edges : Vec<EdgeWithWeight>, verify : bool,
			print : PrintType, allow_multi_edges : bool, iterations: usize) -> Helper {
		let mut h = Helper{ num_vertices, input_edges, input_edge_weights : HashMap::new(), verify,
				print, allow_multi_edges, verification_total_weight : None,
				verification_edges : None ,iterations};
		for (u, v, weight) in &h.input_edges {
			h.input_edge_weights.insert( (*u,*v),*weight );
			h.input_edge_weights.insert( (*v,*u),*weight );
		}
		h
	}
	
	fn get_input_edge_weight( &self, u : usize, v : usize ) -> usize {
		*self.input_edge_weights.get( &(u, v) )
			.expect( format!( "Not an input edge: ({u}, {v})" ).as_str() )
	}
	
	fn compute_total_weight( &self, edges : &Vec<(usize, usize)> ) -> usize {
		edges.iter().map( |(u,v)| self.get_input_edge_weight( *u, *v ) ).sum()
	}
	
	fn report_test_result( &mut self, impl_name : &str, dur : Duration ) {
		if self.print == Print {
			let millis = dur.as_micros() as f64 / 1000.;
			let micros_per_edge = dur.as_micros() as f64 / self.input_edges.len() as f64;
			println!( "{:<20} {millis:10.3}ms ({micros_per_edge:7.3}µs/edge)", impl_name.to_owned() + ":" )
		}
		else if self.print == Json {
			println!( "{}", json::stringify( json::object!{
				num_vertices : self.num_vertices,
				num_edges : self.input_edges.len(),
				name : impl_name,
				time_ns : dur.as_nanos() as usize
			} ) )
		}
	}
	
	fn mst_petgraph( &mut self, benchmark: bool ) {
		// Petgraph MST
		let start = Instant::now();
		let mut g : graph::UnGraph<(), usize> = graph::UnGraph::new_undirected();
		let g_nodes : Vec<graph::NodeIndex> = (0..self.num_vertices).map( |_| g.add_node( () ) ).collect();
		for (u, v, weight) in &self.input_edges {
			g.add_edge( g_nodes[*u], g_nodes[*v], *weight );
		};
		let mst : UnGraph<(), usize> = graph::UnGraph::from_elements( petgraph::algo::min_spanning_tree( &g ) );
		let dur = start.elapsed();
		if benchmark {
			self.report_test_result(  "Kruskal (petgraph)", dur );
		}
		
		let pg_edges = mst.edge_indices()
			.map( |e| mst.edge_endpoints( e ).unwrap() )
				.map( |(u,v)| ( u.index(), v.index() ) ).collect();
		
		self.verification_total_weight = Some( self.compute_total_weight( &pg_edges ) );
		self.verification_edges = Some( normalize_edge_vector( &pg_edges ) );
	}
	
	fn benchmark_mst<TDynForest>( &mut self, impl_name : &str )
		where TDynForest: DynamicForest<TWeight = MSTWeight>
	{
		let mut durations: Vec<Duration> = vec![];
		for _ in 0..self.iterations {

			let start = Instant::now();

			let mut f = TDynForest::new( self.num_vertices );
			let mst = compute_mst( &mut f, self.input_edges.iter().copied(), self.allow_multi_edges );

			let dur = start.elapsed();

			durations.push(dur);

			if self.verify {
				let out_edges = mst.iter().map( |(u,v)| (u.index(), v.index()) ).collect();
				let exp_total_weight = self.verification_total_weight.expect( "When verifying, you must first call mst_petgraph()!" );
				let actual_total_weight = self.compute_total_weight( &out_edges );
				assert_eq!( exp_total_weight, actual_total_weight,
					"Computed incorrect weight, actual: {actual_total_weight}, expected: {exp_total_weight}" );
				assert_eq!( self.verification_edges.as_ref().expect( "When verifying, you must first call mst_petgraph()!" ),
					&normalize_edge_vector( &out_edges ) );
			}
		}
		durations.sort();

		self.report_test_result( impl_name, durations[self.iterations/2] );
	
	}
}


macro_rules! do_benchmark {
	( $obj : ident, $impl_tpl : ident ) => {
		$obj.benchmark_mst::<$impl_tpl<MSTWeight>>( <$impl_tpl<MSTWeight> as ImplName>::name() )
	}
}

fn benchmark( imp : ImplDesc, helper : &mut Helper ) {
	do_for_impl_monoid!( imp, do_benchmark, helper )
}


#[derive(Parser)]
#[command(name = "MST Benchmark")]
struct CLI {
	/// Number of vertices in the underlying graph
	#[arg(short, long, default_value_t = 1_000)]
	num_vertices : usize,
	
	/// Number of edges per vertex in the underlying graph
	#[arg(short, long, default_value_t = 8)] // As in Tarjan, Werneck 2010
	edge_factor : usize,
	
	/// Use complete graph (ignore -e)
	#[arg(long, default_value_t = false)]
	complete : bool,
	
	/// Read input graph from the given file (ignore -n, -e, --complete)
	#[arg(short, long, group = "input")]
	input_file : Option<PathBuf>,
	
	/// Verify the results of each benchmark
	#[arg(long, default_value_t = false)]
	verify : bool,
	
	/// Print the results in human-readable form
	#[arg(long, default_value_t = false)]
	print : bool,
	
	/// Output the results as json
	#[arg(long, default_value_t = false)]
	json : bool,
	
	/// Seed for the random graph generator.
	#[arg(short, long, group = "input")]
	seed : Option<u64>,

	/// Allow inserting edges multiple times (with possibly different weights).
	#[arg(short='m', long, default_value_t = false)]
	allow_multi_edges : bool,
	
	/// Implementations to benchmark. Include all but petgraph if omitted.
	impls : Vec<ImplDesc>,

	/// Number of samples taken when measuring execution time 
	#[arg(long, default_value_t = 1)]
	iterations : usize,
}



fn main() {
	let cli = CLI::parse();
	
	let print = PrintType::from_args( cli.print, cli.json );
	
	let impls : Vec<ImplDesc>;
	if !cli.impls.is_empty() {
		impls = cli.impls;
	}
	else {
		impls = ImplDesc::all_efficient()
	}

	let num_vertices : usize;
	let input_edges : Vec<EdgeWithWeight>;
	
	// Read edges
	if let Some( input_path ) = &cli.input_file {
		if cli.print {
			println!( "Reading edges from '{}'", input_path.display() );
		}
		match read_mst( input_path ) {
			Ok( ( n, e ) ) => { num_vertices = n; input_edges = e },
			Err( e ) => {
				println!( "Could not read file '{}': {}", input_path.display(), e );
				exit( 1 );
			}
		}
		
		if cli.print {
			println!( " Done reading {} edges on {num_vertices} vertices.", input_edges.len() );
		}
	}
	else {
		let seed = cli.seed.unwrap();
		let mut rng = StdRng::seed_from_u64( seed );
		num_vertices = cli.num_vertices;
		let num_edges = num_vertices * cli.edge_factor;
		let all_edges = cli.complete;
		
		// Generate edges
		if all_edges {
			if cli.print {
				println!( "Generating complete graph on {num_vertices} vertices. Seed: {}.", seed );
				stdout().flush().expect( "Couldn't flush for some reason" );
			}
			input_edges = generate_complete_edges_with_weights( num_vertices, &mut rng );
		}
		else {
			if cli.print {
				println!( "Generating sparse random graph on {num_vertices} vertices, with {num_edges} edges. Seed: {}.", seed );
				stdout().flush().expect( "Couldn't flush for some reason" );
			}
			input_edges = generate_sparse_edges_with_weights( num_vertices, num_edges, &mut rng );
		}
		
		if cli.print {
			println!( " Done." );
		}
	}
	
	let mut helper = Helper::new( num_vertices, input_edges, cli.verify, print,
			cli.allow_multi_edges, cli.iterations );
	
	helper.mst_petgraph( true );
	for imp in &impls {
		benchmark( *imp, &mut helper );
	}
}
