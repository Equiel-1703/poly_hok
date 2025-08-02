require PolyHok
use Ske


dev_vet = PolyHok.new_gnx(Nx.tensor(Enum.to_list(1..1000), type: :s32))

x = 1

fun = PolyHok.clo fn y -> x + y end 

host_vet = dev_vet
|> Ske.map(fun)
|> PolyHok.get_gnx

IO.inspect host_vet