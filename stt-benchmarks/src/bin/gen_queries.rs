use std::fmt::{Display, Formatter};
use std::io::{stdout, Write};

use clap::{Parser, ValueEnum};
use num_traits::pow::Pow;
use rand::{distributions, SeedableRng};
use rand::rngs::StdRng;
use stt::common::{EmptyGroupWeight, IsizeAddGroupWeight, UsizeMaxMonoidWeight};
use stt::generate::GeneratableMonoidWeight;

use stt_benchmarks::{bench_util};
use stt_benchmarks::bench_util::{PrintType, Query};
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




struct Helper<TWeight>
	where TWeight : GeneratableMonoidWeight
{
	num_vertices : usize,
	queries : Vec<Query<TWeight>>,
	seed : u64,
	weight_query_prob : f64,
	print : PrintType
}

impl<TWeight> Helper<TWeight>
	where TWeight : GeneratableMonoidWeight
{
	fn new( num_nodes: usize, num_queries : usize, seed : u64, print : PrintType,
		   node_dist : NodeDistribution, weight_query_prob : f64 ) -> Helper<TWeight>
	{
		if print == Print {
			print!( "Generating queries with {node_dist} distribution..." );
			stdout().flush().expect( "Flushing failed!" );
		}
		
		let mut rng = StdRng::seed_from_u64( seed );
		let queries =  match node_dist {
			NodeDistribution::Uniform => bench_util::generate_queries_with_node_dist(
				num_nodes, num_queries, &mut rng, TWeight::generate, weight_query_prob, 
				&distributions::Uniform::new( 0, num_nodes ) ),
			NodeDistribution::Geometric => bench_util::generate_queries_with_node_dist(
				num_nodes, num_queries, &mut rng, TWeight::generate, weight_query_prob,
				&distributions::WeightedIndex::new( 
					(0..num_nodes).map( |i| (1.-GEOM_P).pow( i as f64 ) ) )
					.expect( "Couldn't create distribution" ) )
		};
		
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
	
	/// What weights to use in the benchmark.
	#[arg(short, long, default_value_t = WeightType::Empty)]
	weight : WeightType,
}


fn main() {
	let cli = CLI::parse();
	
	let num_vertices : usize = cli.num_vertices;
	let num_queries = cli.num_queries.unwrap_or( match cli.dist {
		NodeDistribution::Uniform => 20*num_vertices,
		NodeDistribution::Geometric => ( 10. * (1. / (1. - GEOM_P )).pow( num_vertices as f64 ) ) as usize
	} );
	
	let print = PrintType::from_args( cli.print, cli.json );
	
    match cli.weight {
        WeightType::Empty => {
            let helper: Helper<EmptyGroupWeight> = Helper::new( cli.num_vertices,
                num_queries, 
                cli.seed, print, 
                cli.dist, 
                cli.path_query_prob );
            helper.print_queries(); 

        },
        WeightType::Group => {
            let helper: Helper<IsizeAddGroupWeight> = Helper::new( cli.num_vertices,
                num_queries, 
                cli.seed, print, 
                cli.dist, 
                cli.path_query_prob );
            helper.print_queries();

        },
        WeightType::Monoid => {
            let helper: Helper<UsizeMaxMonoidWeight> = Helper::new( cli.num_vertices,
                num_queries, 
                cli.seed, print, 
                cli.dist, 
                cli.path_query_prob );
            helper.print_queries();
            
        }
    } 

    
}