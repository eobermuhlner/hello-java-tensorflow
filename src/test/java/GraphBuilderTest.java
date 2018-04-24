import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import static junit.framework.TestCase.assertEquals;

public class GraphBuilderTest {

	@Test
	public void testConstant () {
		try (Graph g = new Graph()) {
			GraphBuilder builder = new GraphBuilder(g);

			Output<Integer> out = builder.constant("A", 2);

			try (Session s = new Session(g);
				Tensor output = s.runner()
					.fetch(out.op().name())
					.run()
					.get(0)
					.expect(Integer.class)) {
				assertEquals(2, output.intValue());
			}
		}
	}

}
