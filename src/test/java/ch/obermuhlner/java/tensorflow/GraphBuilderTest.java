package ch.obermuhlner.java.tensorflow;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.util.function.Function;

import static junit.framework.TestCase.assertEquals;

public class GraphBuilderTest {

	@Test
	public void testConstant () {
		assertGraphBuilder(2, g -> g.constant("A", 2));
	}

	private <T> void assertGraphBuilder(T expected, Function<GraphBuilder, Output<T>> graphBuilderFunction) {
		try (Graph g = new Graph()) {
			GraphBuilder builder = new GraphBuilder(g);

			Output<T> out = graphBuilderFunction.apply(builder);
			Class<T> type = (Class<T>)expected.getClass();

			try (Session s = new Session(g);
				Tensor<T> output = s.runner()
					.fetch(out.op().name())
					.run()
					.get(0)
					.expect(type)) {
				T actual = toScalar(output, type);
				assertEquals(expected, actual);
			}
		}
	}

	private <T> T toScalar (Tensor<T> tensor, Class<T> type) {
		if (type == Integer.class) {
			return (T) Integer.valueOf(tensor.intValue());
		}
		if (type == Long.class) {
			return (T) Long.valueOf(tensor.longValue());
		}
		if (type == Double.class) {
			return (T) Double.valueOf(tensor.doubleValue());
		}
		if (type == Float.class) {
			return (T) Float.valueOf(tensor.floatValue());
		}
		throw new IllegalArgumentException("Not supported scalar type for tensor: " + type);
	}

}
