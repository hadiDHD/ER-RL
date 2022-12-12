package org.module.eer.exprel.exp;

import kotlin.Pair;
import org.deeplearning4j.rl4j.space.Encodable;
import org.module.eer.model.Module;
import org.module.eer.model.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class ExpState implements Encodable {

    static final int MAX_ELEMENTS = 25;

    static final int OUTPUT_SIZE = MAX_ELEMENTS * (MAX_ELEMENTS - 1) / 2;

    Module[] modules;

    byte[] input;

    ERModel er;

    double reward = 0;

    private ExpState() {
        super();
    }

    public ExpState(ERModel er) {
        super();
        this.er = er;
        input = new byte[MAX_ELEMENTS];

        modules = new Module[er.elements.size()];
        for (int i = 0; i < er.elements.size(); i++) {
            Module m = new Module();
            Element element = er.elements.get(i);
            m.addElement(element);
            modules[i] = m;
            input[i] = (byte) i;
        }
    }

    public void moveItoJ(int i, int j) {
        if (modules[j].isEmpty()) {
            return;
        }
        for (Element e : modules[i].getElementsArray()) {
            input[er.indices.get(e)] = (byte) j;
        }
        modules[j].merge(modules[i]);
        modules[i].getEntities().clear();
        modules[i].getRelationships().clear();
    }

    public List<Integer> getInvalidActions() {
        ArrayList<Integer> invalid = new ArrayList<>();
        for (int i = 0; i < modules.length; i++) {
            Module m = modules[i];
            if (m.isEmpty()) {
                for (int j = 0; j < modules.length; j++) {
                    if (i != j) {
                        invalid.add(getTriangularMatrixIndex(i, j));
                    }
                }
            }
        }
        return invalid;
    }

    public static int getTriangularMatrixIndex(int a, int b) {
        // https://stackoverflow.com/a/27682124
        if (a < b) {
            int temp = b;
            b = a;
            a = temp;
        }
        a--;
        int size = a * (a + 1) / 2;
        return size + b;
    }

    public static Pair<Integer, Integer> getTriangularMatrixRowAndColumn(int index) {
        // https://math.stackexchange.com/a/1417583
        double numerator = Math.sqrt(8 * index + 1) + 1;
        int row = (int) Math.floor(numerator / 2) - 1;
        int column = index - row * (row + 1) / 2;
        if (column > row || column < 0 || row < 0) {
            throw new RuntimeException("WRONG!");
        }
        return new Pair<Integer, Integer>(row + 1, column);
    }

    @Override
    public Encodable dup() {
        ExpState newState = new ExpState();
        newState.er = er;
        newState.input = Arrays.copyOf(input, input.length);
        newState.modules = deepCopyModules();
        newState.reward = reward;
        return newState;
    }

    private Module[] deepCopyModules() {
        Module[] newSet = new Module[modules.length];
        for (int i = 0; i < modules.length; i++) {
            newSet[i] = modules[i].copy();
        }
        return newSet;
    }

    @Override
    public INDArray getData() {
        return Nd4j.create(input, new long[]{input.length}, DataType.INT8);
    }

    @Override
    public boolean isSkipped() {
        return false;
    }

    @Override
    public double[] toArray() {
        return getData().toDoubleVector();
    }

    public void print() {
        int i = 0;
        for (Module m : modules) {
            if (m.isEmpty()) {
                continue;
            }
            i++;
            System.out.println(m.toString(i));
        }
        System.out.println("MQ Index = " + getMQ());
    }

    public double getMQ() {
        double sum = 0.0;
        for (Module m : modules) {
            sum += m.getMQ().getMQ();
        }
        return sum;
    }

}
