require PolyHok

PolyHok.defmodule Map2D do
    defk map_2D_kernel(d_array, resp_array, step,sizex,sizey,f) do

        x = threadIdx.x + blockIdx.x * blockDim.x
        y = threadIdx.y + blockIdx.y * blockDim.y
        offset = x + y * blockDim.x * gridDim.x
    
         id  = step * offset
        #f(id,id)
        if (offset < (sizex*sizey)) do
          resp_array[id] = f(d_array+id,par1,x,y)
        end
      end

      def map_2D(d_array, f) do
        shape = PolyHok.get_shape_gnx(d_array)
        {sizex,sizey,step} =  case shape do
                                 {l,c} -> {l,c,1}
                                 {l,c,step} -> {l,c,step}
                                 x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                               end
        type = PolyHok.get_type(d_array)
     
        resp_array = PolyHok.new_gnx(shape,type)
         #IO.inspect {sizex,sizey,step}
         block_size = 16
         grid_rows = trunc ((sizex + block_size - 1) / block_size)
         grid_cols = trunc ((sizey + block_size - 1) / block_size)
     
     
         PolyHok.spawn(&Map2D.map_2D_kernel/4,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,resp_array,step,sizex,sizey,f])
           resp_array
       end
end


host_array = Nx.tensor([[1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10],
                        [1,2,3,4,5,6,7,8,9,10]])

d_array = PolyHok.new_gnx(host_array)

host_resp = d_array
            |> Map2D.map_2D(PolyHok.phok fn f x -> x+1 end)
            |> PolyHok.get_gnx

IO.inspect host_resp