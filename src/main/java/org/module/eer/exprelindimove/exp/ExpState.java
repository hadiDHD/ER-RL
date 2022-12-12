package org.module.eer.exprelindimove.exp;

import kotlin.Pair;
import org.deeplearning4j.rl4j.space.Encodable;
import org.module.eer.model.ERModel;
import org.module.eer.model.Element;
import org.module.eer.model.Module;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ExpState implements Encodable {

    static final int MAX_ELEMENTS = 25;

    static final int INPUT_SIZE = MAX_ELEMENTS;

    static final int OUTPUT_SIZE = MAX_ELEMENTS;

    ArrayList<Module> modules;

    byte[] input;

    ERModel er;

    double mq = 0;

    private ExpState() {
        super();
    }

    public ExpState(ERModel er) {
        super();
        this.er = er;
        input = new byte[INPUT_SIZE];

        modules = new ArrayList<>(er.elements.size());
        for (int i = 0; i < er.elements.size(); i++) {
            Module m = new Module();
            Element element = er.elements.get(i);
            m.addElement(element);
            modules.add(m);
        }
//        Arrays.fill(input, (byte) -1);
    }

    public boolean moveItoJ(int i, int j) {
        input[i] = (byte) (j + 1);
        if (i == j) {
            return false;
        }
        Module mJ = modules.get(j);
        Element eI = er.elements.get(i);
        mJ.addElement(eI);
        modules.get(i).removeElement(eI);
        return true;
    }

    public Module findModule(int i) {
        for (Module m : modules) {
            if (m.contains(er.elements.get(i))) {
                return m;
            }
        }
        throw new RuntimeException("Module not found");
    }

    public List<Integer> getInvalidActions() {
        ArrayList<Integer> invalid = new ArrayList<>();
//        for (Module m : modules) {
//            if (!m.isEmpty()) {
//                for (Element eI : m.getElementsArray()) {
//                    for (Element eJ : m.getElementsArray()) {
//                        if (eI != eJ) {
//                            invalid.add(getTriangularMatrixIndex(er.indices.get(eI), er.indices.get(eJ)));
//                        }
//                    }
//                }
//            }
//        }
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
        newState.mq = mq;
        return newState;
    }

    private ArrayList<Module> deepCopyModules() {
        ArrayList<Module> newSet = new ArrayList<>(modules.size());
        for (Module module : modules) {
            newSet.add(module.copy());
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
