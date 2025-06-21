require PolyHok
use Ske

defmodule Comp do
    defmacro gpu_for({:<-, _ ,[var,tensor]},do: b)  do
        quote do: Ske.map(unquote(tensor), PolyHok.phok (fn (unquote(var)) -> (unquote b) end))
    end
end