default:
	python human_snake.py --store $(id)

sys:
	for i in $$(seq 1 $(n)); do python systematic_snake.py --store > /dev/null ; done

q:
	for i in $$(seq 1 $(n)); do python q_snake.py play $(p) >> q.txt; done

human:
	for i in $$(seq 1 4); do python human_snake.py --store documentation/humans > /dev/null ; done

evo%:
	python evolve_snake.py play evo_snake-$*.np
