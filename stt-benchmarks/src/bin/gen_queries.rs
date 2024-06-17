use std::fmt::{Display, Formatter};
use std::io::{stdout, Write};

use clap::{Parser, ValueEnum};
use num_traits::pow::Pow;
use rand::distributions::Distribution;
use rand::prelude::SliceRandom;
use rand::{distributions, Rng, SeedableRng};
use rand::rngs::StdRng;
use stt::common::{EmptyGroupWeight, IsizeAddGroupWeight, MonoidWeightWithMaxEdge, UsizeMaxMonoidWeight};
use stt::generate::{generate_edge, generate_edge_with_dist, GeneratableMonoidWeight};

use stt::twocut::splaytt::MonoidTwoPassSplayTT;
use stt::{DynamicForest, MonoidWeight, NodeIdx};
use stt_benchmarks::{bench_util};
use stt_benchmarks::bench_util::{PrintType, Query};
use stt_benchmarks::bench_util::PrintType::*;
use stt_benchmarks::bench_util::Query::*;

const GEOM_P : f64 = 0.01;
const ZIPF_P : f64 = 1.0;

/// A distribution to choose nodes in a dynamic tree
#[derive( Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum )]
enum NodeDistribution {
	Uniform,
	Zipf,
	ZipfQuery
}

impl Display for NodeDistribution {
	fn fmt( &self, f: &mut Formatter<'_> ) -> std::fmt::Result {
		write!( f, "{}", match self {
			Self::Uniform => "uniform",
			Self::Zipf => "power law",
			Self::ZipfQuery => "power law only queries"
		} )
	}
}

/// Enum listing possible weight types.
#[derive( Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum )]
enum WeightType {
	/// No weights and thus no additional data stored per node
	Empty,
	
	/// Signed-add group weights, some strage and update overhead
	Group,
	
	/// Unsigned-max monoid weights, more strage and update overhead
	Monoid
}

impl Display for WeightType {
	fn fmt( &self, f : &mut Formatter<'_> ) -> std::fmt::Result {
		write!( f, "{}", match self {
			Self::Empty => "empty",
			Self::Group => "group",
			Self::Monoid => "monoid"
		} )
	}
}

/// Transforms a list of node pairs into queries
/// 
/// Uses a MonoidTwoPassSplayTT to determine valid queries.
pub fn transform_into_queries<TWeight : MonoidWeight, TRng : Rng>(
	num_nodes : usize,
	node_pairs : impl Iterator<Item=(NodeIdx, NodeIdx)>,
	rng : &mut TRng,
	weight_gen : impl Fn( &mut TRng ) -> TWeight,
	weight_query_prob : f64,
	dist: NodeDistribution)
	-> Vec<Query<TWeight>>
{
	let mut f = MonoidTwoPassSplayTT::<MonoidWeightWithMaxEdge<EmptyGroupWeight>>::new( num_nodes );
	
	node_pairs.map( |(u, v)| {
		if let Some( w ) = f.compute_path_weight(u, v) {
			let (x, y) = w.unwrap_edge();
			if rng.gen_bool( weight_query_prob ) {
				match dist {
					NodeDistribution::ZipfQuery => {
						let mut vertices: Vec<_> = (0..num_nodes).collect();
						vertices[..].shuffle(rng);
						let dist = zipf::ZipfDistribution::new( num_nodes-1, ZIPF_P).unwrap();
		
						PathWeight(NodeIdx::new(vertices[dist.sample(rng)]),NodeIdx::new(vertices[dist.sample(rng)]))
					},

					_ => {
						PathWeight(x, y)
					}
				}

			}
			else {
				f.cut( x, y );
				DeleteEdge( x, y )
			}
		}
		else {
			f.link(u, v, MonoidWeightWithMaxEdge::new( EmptyGroupWeight{}, (u, v) ) );
			InsertEdge(u, v, weight_gen( rng ) )
		}
	} ).collect()
}


struct Helper<TWeight>
	where TWeight : GeneratableMonoidWeight
{
	num_vertices : usize,
	queries : Vec<Query<TWeight>>,
	seed : u64,
	weight_query_prob : f64,
	print : PrintType,
}

impl<TWeight> Helper<TWeight>
	where TWeight : GeneratableMonoidWeight
{
	fn new( num_nodes: usize, num_queries : usize, seed : u64, print : PrintType,
		   node_dist : NodeDistribution, weight_query_prob : f64) -> Helper<TWeight>
	{
		if print == Print {
			print!( "Generating queries with {node_dist} distribution..." );
			stdout().flush().expect( "Flushing failed!" );
		}
		
		let mut rng = StdRng::seed_from_u64( seed );

		//Generate node pairs (correspodns to generate_queries_with_dist_call)
		let node_pairs: Vec<_> = match(node_dist) {
			NodeDistribution::Zipf => {
				let mut vertices: Vec<_> = (0..num_nodes).collect();
				vertices[..].shuffle(&mut rng);
				let dist = zipf::ZipfDistribution::new( num_nodes-1, ZIPF_P).unwrap();
				(0..num_queries)
					.map( |_| generate_edge_with_dist(&dist,&mut rng ) )
					.map( |(u, v)| ( NodeIdx::new( vertices[u] ), NodeIdx::new( vertices[v] ) ) )
					.collect()
			},
			_ =>  {
				let dist = distributions::Uniform::new( 0, num_nodes );
				(0..num_queries)
					.map( |_| generate_edge_with_dist(&dist,&mut rng ) )
					.map( |(u, v)| ( NodeIdx::new( u ), NodeIdx::new( v ) ) )
					.collect()
			}
		};


		let queries = transform_into_queries(num_nodes, node_pairs.into_iter(),&mut rng, TWeight::generate, weight_query_prob,node_dist);
		
		if print == Print {
			println!( " Done." );
		}
		
		Helper{ num_vertices: num_nodes, queries, seed, weight_query_prob, print }
	}

    fn print_queries(&self) {
        for query in &self.queries {
            match query {
                InsertEdge( u, v, weight ) => {
                    println!("i {} {} {}",u,v,weight);
                },
                DeleteEdge( u, v ) => {
                    println!("d {} {}",u,v)
                },
                PathWeight( u, v ) => {
                    println!("p {} {}",u,v)
                },
            }
        }

    }
}

#[derive(Parser)]
#[command(name = "Random query Benchmark")]
struct CLI {
	/// Number of vertices in the underlying graph
	#[arg(short, long, default_value_t = 100)]
	num_vertices : usize,
	
	/// Number of queries
	/// 
	/// [default: 20*NUM_VERTICES for uniform distribution,
	/// 10*(1/0.99)^NUM_VERTICES for geometric distribution]
	#[arg(short='q', long)]
	num_queries : Option<usize>,
	
	/// Probability of generating a path_weight query (instead of a cut) when querying two nodes in
	/// the same tree.
	#[arg(short='p', long, default_value_t = 0.5)]
	path_query_prob : f64,
	
	/// Node distribution to generate queries
	#[arg(short, long, default_value_t = NodeDistribution::Uniform)]
	dist : NodeDistribution,
	
	/// Print the results in human-readable form
	#[arg(long, default_value_t = false)]
	print : bool,
	
	/// Output the results as json
	#[arg(short, long, default_value_t = false)]
	json : bool,
	
	/// Seed for the random query generator
	#[arg(short, long)]
	seed : u64,
}


fn main() {
	let cli = CLI::parse();
	
	let num_vertices : usize = cli.num_vertices;
	let num_queries = cli.num_queries.unwrap_or( match cli.dist {
		_ => 20*num_vertices,
	} );
	
	let print = PrintType::from_args( cli.print, cli.json );
	
	let helper: Helper<EmptyGroupWeight> = Helper::new( 
		cli.num_vertices,
		num_queries, 
		cli.seed, print, 
		cli.dist, 
		cli.path_query_prob
	);
	helper.print_queries(); 
}