
require PolyHok
use Comp

[arg] = System.argv()

m = String.to_integer(arg)

mat1 = Nx.tensor(Enum.to_list(1..(m*m)), type: :f32)
mat2 = Nx.tensor(Enum.to_list(1..(m*m)),  type: :f32)

mat1 = PolyHok.new_gnx(Nx.reshape(mat1,{m,m}))
mat2 = PolyHok.new_gnx(Nx.reshape(mat2,{m,m}))

prev = System.monotonic_time()



result = Comp.gpu_for x <- 0..m, y <- 0..m do
            sum = 0
            for i in range(0,m,1) do
                  sum = sum + mat1[x * m + i] * mat2[i * m + y]
            end
            sum
          end

next = System.monotonic_time()
IO.puts "PolyHok\t#{m}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "

IO.inspect result
