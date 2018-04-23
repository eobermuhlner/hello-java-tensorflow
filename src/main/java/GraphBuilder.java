import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

public class GraphBuilder {

	 private final Graph graph;

	 public GraphBuilder (Graph graph) {
		  this.graph = graph;
	 }

	 public <T> Output<T> constant (String name, Object value, Class<T> type) {
		  try (Tensor<T> t = Tensor.create(value, type)) {
				return graph.opBuilder("Const", name)
					.setAttr("dtype", DataType.fromClass(type))
					.setAttr("value", t)
					.build()
					.output(0);
		  }
	 }

	 public Output<Integer> constant (String name, int value) {
		  return constant(name, value, Integer.class);
	 }

	 public <T> Output<T> placeholder (String name, Class<T> type) {
		  return graph.opBuilder("Placeholder", name)
			  .setAttr("dtype", DataType.fromClass(type))
			  .build()
			  .output(0);
	 }

	 public <T> Output<T> add(Output<T> in1, Output<T> in2) {
	 	 return binaryOp("Add", in1, in2);
	 }

	 public <T> Output<T> sub(Output<T> in1, Output<T> in2) {
		  return binaryOp("Sub", in1, in2);
	 }

	 public <T> Output<T> mul(Output<T> in1, Output<T> in2) {
		  return binaryOp("Mul", in1, in2);
	 }

	 public <T> Output<T> div(Output<T> in1, Output<T> in2) {
		  return binaryOp("Div", in1, in2);
	 }

	 private <T> Output<T> binaryOp(String type, Output<T> in1, Output<T> in2) {
	 	 return graph.opBuilder(type, type)
			 .addInput(in1)
			 .addInput(in2)
			 .build()
			 .output(0);
	 }
}
