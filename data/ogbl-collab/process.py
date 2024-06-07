#!/usr/bin/python3

from typing import *
from typing import TextIO # Bug

import collections
import dataclasses
import os
import random
import sys


@dataclasses.dataclass
class OGBLCollabEdge :
	author1 : int
	author2 : int
	weight : int
	year : int

def load_ogbl_collab( path : str ) -> Iterator[OGBLCollabEdge] :
	try :
		with open( os.path.join( path, "edge.csv" ), "r" ) as fp :
			edge_list = []
			for line in fp :
				x, y = line.split( "," )
				edge_list.append( ( int( x ), int( y ) ) )
		with open( os.path.join( path, "edge_weight.csv" ), "r" ) as fp :
			edge_weight_list = [int( w ) for w in fp]
		with open( os.path.join( path, "edge_year.csv" ), "r" ) as fp :
			edge_year_list = [int( y ) for y in fp]
	except FileNotFoundError as e :
		sys.exit( f"Error: File not found: '{e.filename}'. Have you downloaded the data set? See README.md" )

	assert len( edge_list ) == len( edge_weight_list ) == len( edge_year_list )

	for e, w, y in zip( edge_list, edge_weight_list, edge_year_list ) :
		a1, a2 = e
		yield OGBLCollabEdge( a1, a2, w, y )

def aggregate_ogbl_collab( edges : Iterable[OGBLCollabEdge] ) -> Iterator[OGBLCollabEdge] :
	""" Add weights to subsequent occurrences of the same edge. """
	edge_weights : Dict[(int, int), int] = {}

	for e in sorted( edges, key = lambda e : e.year ) :
		key = (e.author1, e.author2)
		edge_weights[key] = edge_weights.get( key, 0 ) + e.weight
		yield OGBLCollabEdge( e.author1, e.author2, edge_weights[key], e.year )

def as_incremental_msf( edges : Iterable[OGBLCollabEdge], out : TextIO ) -> None :
	print( "Writing incremental MSF..." )
	edges = sorted( edges, key = lambda e : e.year )
	authors = sorted( set( e.author1 for e in edges ).union( e.author2 for e in edges ) )
	author_map = { author : idx for idx, author in enumerate( authors ) }

	max_weight = max( e.weight for e in edges )

	#out.write( "c Generated from ogbl-collab dataset\n" )
	out.write( f"mst {len(authors)} {len(edges)}\n" )

	for e in edges :
		out.write( f"e {author_map[e.author1]} {author_map[e.author2]} {max_weight - e.weight}\n" )


def main() :
	print( "Reading ogbl-collab")
	es = list( load_ogbl_collab( "." ) )
	print( f"{len(es)} edges" )
	heavy_es = [e for e in es if e.weight > 1]
	print( f"{len(heavy_es)} heavy edges" )

	with open( "ogbl-collab.txt", "w" ) as fp :
		as_incremental_msf( aggregate_ogbl_collab( es ), fp )

	print( "Done." )

if __name__ == "__main__" :
	main()
