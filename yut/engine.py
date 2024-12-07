import numpy as np
import copy
import pickle
import unittest.mock
import traceback

import yut.rule



class Player:
	def name(self):
		return "Player"

	def reset(self, random_state):
		pass

	def action(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state
		return mal_to_move, yutscore_to_use, shortcut, debug_msg

	def on_my_action(self, state, my_action, result):
		pass

	def on_enemy_action(self, state, enemy_action, result):
		pass



class EventPrinter:
	def on_game_start(self, name1, name2):
		self.player_names = (name1, name2)

	def on_turn_begin(self, turn):
		self.current_turn = turn
		print( "\n===============Turn #%d: Player %d (%s)'s turn===============\n" % (turn, turn%2+1, self.player_names[turn%2]) )

	def on_yut_cast(self, cast_outcome):
		if len(cast_outcome) > 0:
			print( "\tcast outcome = %s: %s" % (cast_outcome, ", ".join(yut.rule.yutscore_names(cast_outcome))) )


	def on_state(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state
		print( "\tavailable yut scores = %s: %s" % (available_yutscores, ", ".join(yut.rule.yutscore_names(available_yutscores))) )

	def on_action(self, action, result):
		mal_to_move, yutscore_to_use, shortcut, debug_msg = action
		legal_move, my_positions, enemy_positions, num_mals_caught = result

		print( "\taction = %s: move mal %d using '%s' %s shortcut" % ( str((mal_to_move, yutscore_to_use, shortcut)), mal_to_move, yut.rule.yutscore_name(yutscore_to_use), "with" if shortcut else "without" ) )
		if len(debug_msg) > 0:
			print( "\t\tdebug message: '%s'" % debug_msg )

		print( "\tresult:" )
		if self.current_turn%2 == 0:
			yut.rule.print_board_positions( my_positions, enemy_positions )
		else:
			yut.rule.print_board_positions( enemy_positions, my_positions )
		if num_mals_caught > 0:
			print( "\t%d mals catched" % num_mals_caught )
		print()

	def on_error(self, message):
		print( message )

	def on_game_end(self, winner):
		print( "Player %d (%s) won!" % (winner+1, self.player_names[winner]) )


class EventLogger:
	def __init__(self):
		self.player_names = ("", "")
		self.events = []
		self.error_message = None
		self.winner = None

		self.current_turn = None
		self.current_cast_outcome = []
		self.current_state = None

	def on_game_start(self, name1, name2):
		self.__init__()
		self.player_names = (name1, name2)

	def on_turn_begin(self, turn):
		self.current_turn = turn

	def on_yut_cast(self, cast_outcome):
		self.current_cast_outcome.extend( cast_outcome )

	def on_state(self, state):
		self.current_state = state

	def on_action(self, action, result):
		self.events.append( copy.deepcopy( (self.current_turn, self.current_cast_outcome, self.current_state, action, result) )  )
		self.current_cast_outcome = []

	def on_error(self, message):
		self.error_message = message

	def on_game_end(self, winner):
		self.winner = winner

	def print(self):
		printer = EventPrinter()
		printer.on_game_start( self.player_names[0], self.player_names[1] )

		prev_turn = -1
		for turn, cast_outcome, state, action, result in self.events:
			if prev_turn != turn:
				printer.on_turn_begin( turn )
				prev_turn = turn
			printer.on_yut_cast( cast_outcome )
			printer.on_state( state )
			printer.on_action( action, result )

		if self.winner is not None:
			printer.on_game_end( self.winner )

	def save(self, filename):
		with open( filename, "wb" ) as fd:
			pickle.dump( (self.player_names, self.events, self.winner), fd )

	def load(self, filename):
		with open( filename, "rb" ) as fd:
			self.player_names, self.events, self.winner = pickle.load( fd )
		



class GameEngine:
	def __init__(self, raise_exception=True):
		self.raise_exception = raise_exception

	def reset(self, player1, player2, seed):
		self.player1 = player1
		self.player1_positions = (0,0,0,0)
		self.player1_random = np.random

		self.player2 = player2
		self.player2_positions = (0,0,0,0)
		self.player2_random = np.random

		if seed is None:
			self.player1_random = np.random
			self.player2_random = np.random

			player1.reset( np.random )
			player2.reset( np.random )

		else:
			np.random.seed(seed)
			self.player1_random = np.random.RandomState(seed+1)
			self.player2_random = np.random.RandomState(seed+2)

			player1.reset( np.random.RandomState(seed+1) )
			player2.reset( np.random.RandomState(seed+2) )

	def play_single_turn(self, turn, event_listener, game_states, game_actions):
		if turn%2 == 0:
			me, enemy = self.player1, self.player2
			random_state = self.player1_random
			my_positions, enemy_positions = self.player1_positions, self.player2_positions
		else:
			me, enemy = self.player2, self.player1
			random_state = self.player2_random
			my_positions, enemy_positions = self.player2_positions, self.player1_positions
		

		event_listener.on_turn_begin( turn )

		cast_outcome = yut.rule.random_cast( random_state )
		available_yutscores = cast_outcome

		event_listener.on_yut_cast( cast_outcome )

		while len(available_yutscores) > 0:
			state = (turn, my_positions, enemy_positions, available_yutscores)
	
			if game_states is not None:
				game_states.append( (my_positions , enemy_positions, tuple(available_yutscores)) )
   
			event_listener.on_state( state )
			legal_move = True

			try:
				action = mal_to_move, yutscore_to_use, shortcut, debug_msg = me.action( copy.deepcopy(state) )
				# print( "***action:", action )
			except Exception as exc:
				traceback.print_exc()
				legal_move = False
				event_listener.on_error( str(exc) )
				break
			
			if game_actions is not None:
				game_actions.append( action )
    
			# skip the turn if there's no available legal action
			if debug_msg == "skip":
				break

			if yutscore_to_use not in available_yutscores:
				print("unavailable yutscore is used")
				result = legal_move, my_positions, enemy_positions, num_mals_caught = False, my_positions, enemy_positions, 0
			else:
				result = legal_move, my_positions, enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mal_to_move, yutscore_to_use, shortcut )	

			event_listener.on_action( action, result )

			me.on_my_action( (turn, my_positions, enemy_positions, available_yutscores), action, (legal_move, my_positions, enemy_positions, num_mals_caught) )
			enemy.on_enemy_action( (turn, enemy_positions, my_positions, available_yutscores), action, (legal_move, enemy_positions, my_positions, num_mals_caught) )

			if legal_move == False:
				print( "ALERT: Player %d (%s) made an illegal move on turn #%d" % (turn%2+1, me.name(), turn) )
				print( "\t\tstate=", state, ", action=", action )
				event_listener.on_error( "Player %d made an illegal move" % (turn%2+1) )
				break

			available_yutscores.remove( yutscore_to_use )
			if num_mals_caught > 0 and yut.rule.needs_throw_again( yutscore_to_use ) == False:
				cast_outcome = yut.rule.random_cast( random_state )
				available_yutscores.extend( cast_outcome )

				event_listener.on_yut_cast( cast_outcome )
			else:
				cast_outcome = []
					
			if yut.rule.game_finished( my_positions ):
				break

		if turn%2 == 0:
			self.player1_positions, self.player2_positions = my_positions, enemy_positions
		else:
			self.player2_positions, self.player1_positions = my_positions, enemy_positions

		if legal_move == False:
			return 1 - turn%2
		elif yut.rule.game_finished( my_positions ):
			return turn%2
		return None

	def play(self, player1, player2, game_states=None,game_actions=None, seed=None, game_event_listener=None ):
		self.reset( player1, player2, seed )

		if game_event_listener is None:
			game_event_listener = unittest.mock.Mock()
		game_event_listener.on_game_start( player1.name(), player2.name() )
		
		turn = 0
		while True:
			winner = self.play_single_turn( turn, game_event_listener, game_states, game_actions )
			if winner is not None:
				break
			turn += 1

		game_event_listener.on_game_end( winner )
		return winner


