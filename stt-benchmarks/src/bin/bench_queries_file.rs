use std::fmt::{Display, Formatter};
use std::fs::File;
use std::io::{self, stdout, BufRead, Empty, Write};
use std::num::ParseIntError;
use std::path::PathBuf;
use std::process::exit;
use std::time::Duration;

use clap::{Parser, ValueEnum};
use num_traits::pow::Pow;
use rand::{distributions, SeedableRng};
use rand::rngs::StdRng;
use stt::common::{EmptyGroupWeight, IsizeAddGroupWeight, UsizeMaxMonoidWeight};
use stt::mst::EdgeWithWeight;
use stt::DynamicForest;
use stt::generate::GeneratableMonoidWeight;
use stt::link_cut::*;
use stt::onecut::*;
use stt::pg::*;
use stt::twocut::mtrtt::*;
use stt::twocut::splaytt::*;
use stt::NodeIdx;

use stt_benchmarks::{bench_util, do_for_impl_empty, do_for_impl_group, do_for_impl_monoid};
use stt_benchmarks::bench_util::{ImplDesc, ImplName, PrintType, Query};
use stt_benchmarks::bench_util::PrintType::*;
use stt_benchmarks::bench_util::Query::*;

const GEOM_P : f64 = 0.01;

/// A distribution to choose nodes in a dynamic tree
#[derive( Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum )]
enum NodeDistribution {
	Uniform,
	
	/// Geometric distribution with p=[GEOM_P]
	Geometric
}

impl Display for NodeDistribution {
	fn fmt( &self, f: &mut Formatter<'_> ) -> std::fmt::Result {
		write!( f, "{}", match self {
			Self::Uniform => "uniform",
			Self::Geometric => "geometric"
		} )
	}
}


struct Helper
{
	num_vertices : usize,
	queries : Vec<Query<EmptyGroupWeight>>,
	print : PrintType,
	iterations : usize
}

impl Helper
{
	fn new( num_nodes: usize, print : PrintType, queries: Vec<Query<EmptyGroupWeight>>, iterations:usize) -> Helper
	{	
		Helper{ num_vertices: num_nodes, queries, print, iterations }
	}
	
	fn benchmark<TDynForest>( &self, impl_name : &str)
		where TDynForest : DynamicForest<TWeight=EmptyGroupWeight>
	{
		let iterations = self.iterations;
		let mut durations: Vec<Duration> = vec![];
		for _ in 0..iterations {
			durations.push(bench_util::benchmark_queries::<TDynForest>( self.num_vertices, &self.queries ));

		}
		durations.sort();
		let duration = durations[iterations/2];

		if self.print == Print {
			let per_query_str = format!( "({:.3}µs/query)", duration.as_micros() as f64 / ( self.queries.len() as f64 ) );
			println!( "{impl_name:<20} {:8.3}ms {per_query_str:>17}", duration.as_micros() as f64 / 1000. )
		}
		else if self.print == Json {
			println!( "{}", json::stringify( json::object!{
				name : impl_name,
				num_vertices : self.num_vertices,
				num_queries : self.queries.len(),
				time_ns : usize::try_from( duration.as_nanos() )
					.expect( format!( "Duration too long: {}", duration.as_nanos() ).as_str() )
			} ) )
		}
	}
	
	fn print_query_type_dist( &self ) {
		let mut inserts = 0;
		let mut deletes = 0;
		let mut path_weights = 0;
		
		for query in &self.queries {
			match query {
				Query::InsertEdge( _, _, _ ) => inserts += 1,
				Query::DeleteEdge( _, _ ) => deletes += 1,
				Query::PathWeight( _, _ ) => path_weights +=1
			}
		}
		
		println!( "Generated {inserts}x Link, {deletes}x Cut, {path_weights}x PathWeight" );
	}
}





fn benchmark( helper : &Helper, impls: &Vec<ImplDesc> ) {
	if helper.print == Print {
		println!( "Benchmarking {} connectivity queries on {} vertices",
			helper.queries.len(), helper.num_vertices );
		helper.print_query_type_dist();
	}
	
	macro_rules! do_benchmark_empty {
		( $obj : ident, $impl_tpl : ident ) => {
			$obj.benchmark::<$impl_tpl>( <$impl_tpl as ImplName>::name() )
		}
	}
	
	for imp in impls {
		do_for_impl_empty!( imp, do_benchmark_empty, helper );
	}
}

fn read_queries( path : &PathBuf ) -> io::Result<(usize, Vec<Query<EmptyGroupWeight>>)> {
	//Open path and initialize n, query vector.
	let file = File::open( path )?;
	let mut num_vertices = 0;
	let mut edges : Vec<Query<EmptyGroupWeight>> = vec![];

	//For every line, 
	for line in io::BufReader::new( file ).lines() {
		let line = line?;
		let parts : Vec<_> = line.split( " " ).collect();

		//Case 1: First line.
		if parts[0] == "con" {
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

		fn parse_edge( edge_parts : &Vec<&str>) -> Result<(NodeIdx,NodeIdx), ParseIntError> {
			let u = NodeIdx::new(edge_parts[1].parse()?);
			let v = NodeIdx::new(edge_parts[2].parse()?);
			return Ok( (u,v) )
		}

		//Case 2: "_ <from> <to> <empty>"
		if parts.len() > 4 {
			return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
		} 
		let Ok((u,v)) = parse_edge(&parts) else {
			return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
		};


		let q: Query<EmptyGroupWeight> = match parts[0] {
			"i" =>{
				InsertEdge(u,v,EmptyGroupWeight{})
			},
			"d" =>{
				DeleteEdge(u,v)
			},
			"p" =>{
				PathWeight(u,v)
			},
			_ => {
				return Err( io::Error::new( io::ErrorKind::Other, format!( "Invalid line: '{line}'" ) ) );
			}
		};
		edges.push(q);
	}
	Ok( (num_vertices, edges) )
}


#[derive(Parser)]
#[command(name = "Random query Benchmark")]
struct CLI {
	/// Node distribution to generate queries
	#[arg(short, long, default_value_t = 1)]
	iterations : usize,

	/// Node distribution to generate queries
	#[arg(short, long, default_value_t = NodeDistribution::Uniform)]
	dist : NodeDistribution,

	/// Read input graph from the given file (ignore -n, -e, --complete)
	#[arg(short, long, group = "input")]
	input_file : Option<PathBuf>,
	
	/// Print the results in human-readable form
	#[arg(short, long, default_value_t = false)]
	print : bool,
	
	/// Output the results as json
	#[arg(short, long, default_value_t = false)]
	json : bool,
	
	/// Implementations to benchmark. Include all but petgraph if omitted.
	impls : Vec<ImplDesc>
}


fn main() {
	let cli = CLI::parse();
	
	let print = PrintType::from_args( cli.print, cli.json );
	
	let num_vertices : usize;
	let input_queries : Vec<Query<EmptyGroupWeight>>;

	if let Some( input_path ) = &cli.input_file {
		match read_queries( input_path ) {
			Ok( ( n, e ) ) => { num_vertices = n; input_queries = e },
			Err( e ) => {
				println!( "Could not read file '{}': {}", input_path.display(), e );
				exit( 1 );
			}
		}
		
		if cli.print {
			println!( "Done reading {} edges on {num_vertices} vertices.", input_queries.len() );
		}
		let impls : Vec<ImplDesc>;
	
		if !cli.impls.is_empty() {
			impls = cli.impls;
		}
		else {
			impls = ImplDesc::all_efficient()
		}
		
		let helper = &Helper::new(num_vertices, print, input_queries,cli.iterations);

		benchmark( helper, &impls );
	}

}
