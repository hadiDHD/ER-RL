package old;

import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class ErState implements Encodable {

    byte[] data;

    public ErState(int size) {
        data = new byte[size * 2 - 1];
        for (int i = 0; i < size; i++) {
            // even places are element indices
            data[i * 2] = (byte) i;
        }
        for (int i = 0; i < size - 1; i++) {
            // odd places are separators
            data[i * 2 + 1] = 0;
        }
    }

    public void swap(int i, int j) {
        if (i % 2 != 0 || j % 2 != 0) {
            throw new RuntimeException("Cannot swap odd places!");
        }
        byte temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }

    public void toggle(int i) {
        if (i % 2 == 0) {
            throw new RuntimeException("Cannot toggle even places!");
        }
        data[i] = (byte) (1 - data[i]);
    }

    @Override
    public double[] toArray() {
        return getData().toDoubleVector();
    }

    @Override
    public boolean isSkipped() {
        return data == null;
    }

    @Override
    public INDArray getData() {
        return Nd4j.create(data, new long[]{data.length}, DataType.INT8);
    }

    @Override
    public Encodable dup() {
        return null;
    }
}
