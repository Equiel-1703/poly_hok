require PolyHok
use Ske


dev_vet = PolyHok.new_gnx(Nx.tensor(Enum.to_list(1..1000), type: :f32))

x = 10

fun = PolyHok.clo fn y -> x + y end 

host_vet = dev_vet
|> Ske.map(fun)
|> PolyHok.get_gnx

IO.inspect host_vet