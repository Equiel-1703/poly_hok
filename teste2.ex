require PolyHok

use Ske

PolyHok. defmodule Joao do

defd succ(x) do

x + 1 

end

defd mult (x) do

× *2

end end

n = 1000000

arr = Nx. tensor (Enum.to_list(1..n), type: {:f, 64})

## Benchmarking

tO = System. monotonic_time()

res = arr

|> PolyHok.new_gnx()

|> Ske. map (PolyHok. phok fn x →>

type double r1

r1 = succ (x)

mult (r1)

end)

|> PolyHok. get_gnx()

t1 = System monotonic_time()

elapsed_ms = System. convert_time_unit(t1-to, native, :millisecond)

Io.puts （"PolyHoklt#in｝lt#felapsed

_ms}" )

IO. inspect (res)