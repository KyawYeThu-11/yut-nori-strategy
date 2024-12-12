import numpy as np
import scipy.stats
import yut.engine

distance_to_goal = np.zeros( yut.rule.FINISHED+1 )
outcomes, probs = yut.rule.enumerate_all_cast_outcomes(depth=5)

for _ in range(10):
	for s in range( yut.rule.FINISHED-1, -1, -1):
		weighted_sum = 0.0
		for outcome, prob in zip( outcomes, probs ):
			pos = s
			for ys in outcome:
				pos = yut.rule.next_position( pos, ys, True )
			weighted_sum += ( 1 + distance_to_goal[pos] ) * prob
		distance_to_goal[s] = weighted_sum

def evaluate_score( my_positions, enemy_positions, throw_again ):
	my_duplicates = [ sum(np == p for np in my_positions) for p in my_positions ]
	enemy_duplicates = [ sum(np == p for np in enemy_positions) for p in enemy_positions ]
	multipliers = [ 1, 1, 0.7, 0.4, 0.3 ]

	return - sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(my_positions,my_duplicates) ) \
			+ sum( distance_to_goal[p] * (multipliers[np] if p != 0 else 1) for p,np in zip(enemy_positions,enemy_duplicates) ) \
			+ ( +1 if throw_again else 0 )


class ExamplePlayer(yut.engine.Player):
	def name(self):
		return "Example"

	def action(self, state):
		turn, my_positions, enemy_positions, available_yutscores = state

		scores = []
		for mi, mp in enumerate(my_positions):
			if mp == yut.rule.FINISHED:
				continue
			for ys in available_yutscores:
				for shortcut in [True, False]:
					legal_move, next_my_positions, next_enemy_positions, num_mals_caught = yut.rule.make_move( my_positions, enemy_positions, mi, ys, shortcut )
					if legal_move:
						scores.append( (evaluate_score(next_my_positions, next_enemy_positions, num_mals_caught>0), mi, ys, shortcut ) )
		scores.sort(reverse=True)

		return scores[0][1], scores[0][2], scores[0][3], ""


if __name__ == "__main__":
	p = ExamplePlayer()
	engine = yut.engine.GameEngine()
	for s in range(100):
		winner = engine.play( p, p, seed=s )
		if winner == 0:
			print( "Player 1 won!" )
		else:
			print( "Player 2 won!" )



	