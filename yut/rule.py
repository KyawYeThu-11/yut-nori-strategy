import numpy as np
import scipy.stats

FLATSIDEUP_PROB = 0.6

YUTSCORES = ( 1, 2, 3, 4, 5, -1 )
YUTSCORE_NAMES = { 1:"do", 2:"gae", 3:"geol", 4:"yut", 5:"mo", -1:"backdo" }
N_MALS = 4

GRID = (
( 10,  9,  8, -1,  7,  6,  5 ),
( 18, 11, -1, -1, -1, 13,  4 ),
( 19, -1, 12, -1, 14, -1,  3 ),
( -1, -1, -1, 15, -1, -1, -1 ),
( 20, -1, 16, -1, 23, -1,  2 ),
( 21, 17, -1, -1, -1, 24,  1 ),
( 22, 25, 26, -1, 27, 28, 29 ) )
FINISHED = 30

def needs_throw_again( yutscore ):
	"""Return whether the player can throw yuts again"""
	return yutscore >= 4

def game_finished( positions ):
	"""Return whether all mals have finished running the board"""
	return min(positions) == FINISHED


def yutscore_name( yutscore ):
	return YUTSCORE_NAMES[ yutscore ]

def yutscore_names( yutscores ):
	return [ YUTSCORE_NAMES[y] for y in yutscores ]


_prob_yutscores = np.zeros(6)
_prob_yutscores[:5] = scipy.stats.binom.pmf( [1,2,3,4,0], 4, FLATSIDEUP_PROB )
_prob_yutscores[0], _prob_yutscores[-1] = _prob_yutscores[0]*0.75, _prob_yutscores[0]*0.25


def _next_position(prev, cur, yutscore, shortcut=False):
	if yutscore == 0:
		return cur
	default_next = {	0:1,
						1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:18, 
						11:12, 12:15, 13:14, 14:15, 15:16, 16:17, 17:22, 18:19, 19:20, 20:21,
						21:22, 22:25, 23:24, 24:29, 25:26, 26:27, 27:28, 28:29, 29:30, 30:30 }
	if yutscore == -1:
		if cur == FINISHED:
			return FINISHED
		elif cur == 0:
			return 0
		elif cur == 1:
			return 29
		elif cur == 15:
			return 14 if shortcut else 12
		elif cur == 22:
			return 17 if shortcut else 21
		elif cur == 29:
			return 24 if shortcut else 28

		elif cur == 23:
			return 15
		elif cur == 11:
			return 10
		elif cur == 13:
			return 5

		return sorted( [ p for p,n in default_next.items() if n==cur ] )[-1]

	if cur == 15 and (prev == -1 or prev == 12):
		return _next_position(cur, 23, yutscore-1)
	elif cur == 10 and prev == -1:
		return _next_position(cur, 11, yutscore-1)
	elif cur == 5 and prev == -1:
		return _next_position(cur, 13, yutscore-1)

	return _next_position(cur, default_next[cur], yutscore-1)


_nextp_map = np.zeros( (FINISHED+1, len(YUTSCORES)+1, 2), dtype=np.int32 )
for p in range(FINISHED+1):
	for y in YUTSCORES:
		_nextp_map[p][y][0] = _next_position( -1, p, y, 0 )
		_nextp_map[p][y][1] = _next_position( -1, p, y, 1 )

def next_position( current_position, yutscore, shortcut ):
	return _nextp_map[ current_position ][ yutscore ][ int(shortcut) ]


def random_cast(random_state=np.random):
	cast_outcome = [ random_state.choice( YUTSCORES, p=_prob_yutscores ) ]
	while needs_throw_again(cast_outcome[-1]):
		cast_outcome.append( random_state.choice( YUTSCORES, p=_prob_yutscores ) )
	return cast_outcome

def enumerate_all_cast_outcomes(depth=1):
	if depth == 1:
		return [ [y] for y in YUTSCORES ], _prob_yutscores
	outcomes, probs = [], []
	inner_outcomes, inner_probs = enumerate_all_cast_outcomes(depth-1)
	for y,p in zip(YUTSCORES, _prob_yutscores):
		if needs_throw_again(y) == False:
			outcomes.append( [y] )
			probs.append( p )
		else:
			for io,ip in zip(inner_outcomes, inner_probs):
				outcomes.append( [y]+io )
				probs.append( p*ip )

	return outcomes, probs

def make_move(my_positions, enemy_positions, mal_to_move, yutscore, shortcut ):
	curp = my_positions[ mal_to_move ]

	if curp == FINISHED:	# trying to move a mal who finished the run
		return False, my_positions, enemy_positions, 0
	if curp == 0 and yutscore == -1 and any(0 < p < FINISHED for p in my_positions):		# trying to move a mal backward
		return False, my_positions, enemy_positions, 0
		
	nextp = next_position( curp, yutscore, shortcut )

	if curp == 0:
		mals_to_move = [ mal_to_move ]
	else:
		mals_to_move = [ i for i in range(N_MALS) if my_positions[i] == curp ]

	if nextp == FINISHED or nextp == 0:
		return True, tuple(nextp if mi in mals_to_move else mp for mi,mp in enumerate(my_positions)), enemy_positions, 0
	else:
		return True, tuple(nextp if mi in mals_to_move else mp for mi,mp in enumerate(my_positions)), tuple(0 if ep == nextp else ep for ep in enemy_positions), sum(ep == nextp for ep in enemy_positions)


def print_board_custom( annotation = lambda position:"", blank=0, extra=[], indentation="" ):
	for i in range(7):
		row = [ indentation ]
		for j in range(7):
			pos = GRID[i][j]
			if pos == -1:
				row.append( " "*(6+blank) )
			else:
				string_format = "[%%2d:%%-%ds] " % blank
				row.append( string_format%(pos, annotation(pos)) )

		if i >= 7-len(extra):
			row.append( extra[i-(7-len(extra))] )
		print( "".join(row) )



def print_board_positions( player1_positions, player2_positions, player1_name="Player 1", player2_name="Player 2", indentation="" ):
	print_board_custom( 
		annotation = lambda p: "".join("abcd"[mi] for mi,mp in enumerate(player1_positions) if mp==p) + "".join("ABCD"[mi] for mi,mp in enumerate(player2_positions) if mp==p), 
		blank=4, 
		extra = [ "%s=%s" % ( player1_name, str(tuple(player1_positions)) ), "%s=%s" % ( player2_name, str(tuple(player2_positions)) ) ], 
		indentation=indentation )

