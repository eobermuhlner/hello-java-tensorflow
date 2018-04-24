package ch.obermuhlner.java.tensorflow;

import org.tensorflow.*;

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

	public <T> Output<T> variable (String name, Class<T> type, Shape shape) {
		return graph.opBuilder("Variable", name)
		   .setAttr("dtype", DataType.fromClass(type))
		   .setAttr("shape", shape)
		   .build()
		   .output(0);
	}

	public <T> Output<T> assign (Output<T> variable, Output<T> value) {
		return op("Assign", "Assign/" + variable, variable, value);
	}

	public <T> Output<T> add(Output<T> in1, Output<T> in2) {
	 	 return op("Add", in1, in2);
	 }

	 public <T> Output<T> sub(Output<T> in1, Output<T> in2) {
		  return op("Sub", in1, in2);
	 }

	 public <T> Output<T> mul(Output<T> in1, Output<T> in2) {
		  return op("Mul", in1, in2);
	 }

	 public <T> Output<T> div(Output<T> in1, Output<T> in2) {
		  return op("Div", in1, in2);
	 }

	private <T> Output<T> op(String type, Output<T> in1) {
		return graph.opBuilder(type, type)
		   .addInput(in1)
		   .build()
		   .output(0);
	}

	private <T> Output<T> op(String type, Output<T> in1, Output<T> in2) {
		return op(type, type, in1, in2);
	}

	private <T> Output<T> op(String type, String name, Output<T> in1, Output<T> in2) {
		return graph.opBuilder(type, name)
		   .addInput(in1)
		   .addInput(in2)
		   .build()
		   .output(0);
	}
}
