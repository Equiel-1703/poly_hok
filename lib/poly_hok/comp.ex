require PolyHok

PolyHok.defmodule Comp do
    defmacro gpu_for({:<-, _ ,[var,tensor]},do: b)  do
        quote do: Comp.map(unquote(tensor), PolyHok.phok (fn (unquote(var)) -> (unquote b) end))
    end
    def map(input, f) do
        shape = PolyHok.get_shape(input)
        type = PolyHok.get_type(input)
        result_gpu = PolyHok.new_gnx(shape,type)
        size = Tuple.product(shape)
        threadsPerBlock = 128;
        numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)
    
        PolyHok.spawn(&Ske.map_ker/4,
                  {numberOfBlocks,1,1},
                  {threadsPerBlock,1,1},
                  [input,result_gpu,size, f])
        result_gpu
      end
      defk map_ker(a1,a2,size,f) do
        index = blockIdx.x * blockDim.x + threadIdx.x
        stride = blockDim.x * gridDim.x
  
        for i in range(index,size,stride) do
              a2[i] = f(a1[i])
        end
      end
end