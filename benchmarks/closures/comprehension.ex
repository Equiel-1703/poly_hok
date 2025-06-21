require PolyHok
require Comp


dev_vet = PolyHok.new_gnx(Nx.tensor([1,2,3,4,5,6,7,8,9,10]))

Comp.gpu_for n <- dev_vet,  do: n * n
|> PolyHok.get_gnx
|> IO.inspect