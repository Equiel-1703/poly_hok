require PolyHok

use Ske

PolyHok.defmodule Joao do
  defd succ(x) do
    x + 1
  end

  defd mult(x) do
    x * 2
  end

end

n = 1000000
arr = Nx.tensor(Enum.to_list(1..n), type: {:f, 32})

## Benchmarking
t0 = System.monotonic_time()
res = 
  arr
  |> PolyHok.new_gnx()
  |> Ske.map(PolyHok.phok fn x -> 
              type r1 float
              r1 = succ(x)
              mult(r1)
            end) 
  |> PolyHok.get_gnx()

t1 = System.monotonic_time()
elapsed_ms = System.convert_time_unit(t1-t0, :native, :millisecond)
IO.puts("PolyHok\t#{n}\t#{elapsed_ms}")
IO.inspect(res)
