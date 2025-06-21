require PolyHok
use Comp


dev_vet = PolyHok.new_gnx(Nx.tensor([1,2,3,4,5,6,7,8,9,10]))

v = 3

resp = Comp.gpu_for n <- dev_vet,  do: v * n
resp
|> PolyHok.get_gnx
|> IO.inspect