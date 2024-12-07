import numpy as np
import yut.engine


class InteractivePlayer(yut.engine.Player):
	def name(self):
		return "Interactive"

	def action(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state
		print( "Turn #%d (Your turn)" % turn )
		print( "\tCurrent board state:")
		yut.rule.print_board_positions( my_positions, enemy_positions, "You", "Opponent", indentation="\t" )

		available_yutscore_names = yut.rule.yutscore_names(available_yutscores)
		print( "\tCurrently available yut scores:", available_yutscore_names )

		if len( available_yutscores ) > 1:
			while True:
				yutscore_name = input( "\tWhich yut score do you want to use? Choose one out of %s: " % str( available_yutscore_names ) )
				if yutscore_name in available_yutscore_names:
					yutscore_to_use = available_yutscores[ available_yutscore_names.index( yutscore_name ) ]
					break
		else:
			yutscore_to_use = available_yutscores[0]
			yutscore_name = available_yutscore_names[0]

		available_mals = [ chr(i+ord('a')) for i,p in enumerate(my_positions) if p != yut.rule.FINISHED ]
		while True:
			mal = input( "\tWhich mal do you want to move using '%s'? Choose one out of %s: " % ( yutscore_name, str( available_mals ) ) )
			if mal in available_mals:
				mal_to_move = ord(mal) - ord('a')
				break

		curp = my_positions[ mal_to_move ]
		if yut.rule.next_position( curp, yutscore_to_use, False ) != yut.rule.next_position( curp, yutscore_to_use, True ):
			while True:
				yesorno = input( "\tDo you want to use shortcut? Choose one out of ['y', 'n']: " )
				if yesorno == 'y':
					shortcut = True
					break
				elif yesorno == 'n':
					shortcut = False
					break
		else:
			shortcut = True

		return mal_to_move, yutscore_to_use, shortcut, ""


	def on_my_action(self, state, my_action, result):
		legal_move, my_positions, enemy_positions, num_mals_caught = result

		print( "\tResult:" )
		yut.rule.print_board_positions( my_positions, enemy_positions, "You", "Opponent", indentation="\t" )

		if num_mals_caught > 0:
			print( "\tNumber of opponent's mal(s) caught by you: %d"%num_mals_caught )


	def on_enemy_action(self, state, enemy_action, result):
		turn, my_positions, enemy_positions, available_yutscores = state
		mal_to_move, yutscore_to_use, shortcut, debug_msg = enemy_action
		legal_move, my_positions, enemy_positions, num_mals_caught = result
		print( "Turn #%d (Opponent's turn)" % turn )
		print( "\tOpponent moved the mal '%s' using '%s' out of %s" % ( chr(mal_to_move+ord('A')), yut.rule.yutscore_name( yutscore_to_use ), yut.rule.yutscore_names( available_yutscores ) ) )
		print( "\tResult:" )
		yut.rule.print_board_positions( my_positions, enemy_positions, "You", "Opponent", indentation="\t" )
		if num_mals_caught > 0:
			print( "\tNumber of your mal(s) caught by opponent: %d"%num_mals_caught )



if __name__ == "__main__":
	import example_player, sys

	if sys.argv[-1].isdigit():
		seed = int(sys.argv[-1])
	else:
		seed = None
	
	game = yut.engine.GameEngine()
	winner = game.play( player1=InteractivePlayer(), player2=example_player.ExamplePlayer(), seed=seed )

	if winner == 0:
		print( "You won!" )
	else:
		print( "You lose!" )
