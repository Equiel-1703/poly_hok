require PolyHok

PolyHok.defmodule TestBitOp do
  defk map_ker(a1,res,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id, size, stride) do
      res[i] = f(a1[i])
    end
  end

  def map(a1,func) do
    {l,c} = PolyHok.get_shape_gnx(a1)
    type = PolyHok.get_type_gnx(a1)
    size = l*c
    result_gpu = PolyHok.new_gnx(l,c, type)

    threadsPerBlock = 32
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(
      &TestBitOp.map_ker/4,
      {numberOfBlocks,1,1},
      {threadsPerBlock,1,1},
      [a1,result_gpu,size,func]
    )

    result_gpu
  end
end

array_nx = Nx.tensor([2,4,6,8,10,12,14,16,18,20], type: {:s, 32})
IO.inspect(array_nx, label: "Original Nx array")

array_gpu = array_nx |> PolyHok.new_gnx

result_nx = TestBitOp.map(array_gpu, PolyHok.phok fn (x) -> x <<< 1 end) |> PolyHok.get_gnx

IO.inspect(result_nx, label: "Result of bitwise left shift (<<<)")
