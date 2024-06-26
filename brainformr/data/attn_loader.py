"""
	
TODO:
	* need to:
		1. load all cells by genes 
		2. two outputs: n_cells * genes long vec of integers for each gene and then 
			separately the float valued directions
			maybe pretty low dim for this first embed, but who knows
		3. then need to aggregate via pooling or averaging, whatever
		4. how do we agg in this way?
		5. 

		I figured that part out.

		Now I need to find a way to extract for each cell:

		(gene_ids, expr) where gene_ids is a list of integers and expr is the scalars

		i do this at encoding time, now i need a fn to do it at decoding time

"""