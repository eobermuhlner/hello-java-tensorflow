import org.tensorflow.*;

public class HelloTensorflow {
	 public static void main (String[] args) {
	 	 simpleSum();
	 	 placeholderSum(3, 4);
	 }

	 public static void simpleSum () {
		  try (Graph g = new Graph()) {
		  		GraphBuilder builder = new GraphBuilder(g);

		  		Output<Integer> out = builder.add(
		  			builder.constant("A", 2),
					builder.constant("B", 3));

				try (Session s = new Session(g);
					Tensor output = s.runner()
						.fetch(out.op().name())
						.run()
						.get(0)
						.expect(Integer.class)) {
					 System.out.println(output.intValue());
				}
		  }
	 }

	 public static void placeholderSum(int x, int y) {
		  try (Graph g = new Graph()) {
				GraphBuilder builder = new GraphBuilder(g);

				Output<Integer> out = builder.add(
					builder.placeholder("x", Integer.class),
					builder.placeholder("y", Integer.class));

				try (Session s = new Session(g);
					Tensor tensorX = Tensor.create(x, Integer.class);
					Tensor tensorY = Tensor.create(y, Integer.class);
					Tensor output = s.runner()
						.feed("x", tensorX)
						.feed("y", tensorY)
						.fetch(out.op().name())
						.run()
						.get(0)
						.expect(Integer.class)) {
					 System.out.println(output.intValue());
				}
		  }
	 }
}
