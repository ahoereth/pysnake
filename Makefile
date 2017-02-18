default:
	python human_snake.py --store $(id)

sys:
	for i in $$(seq 1 $(n)); do python systematic_snake.py --store > /dev/null ; done

