package org.module.eer.general;

import java.util.*;

import org.deeplearning4j.rl4j.space.Encodable;
import org.module.eer.model.Element;
import org.module.eer.model.Entity;
import org.module.eer.model.Relationship;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import kotlin.Pair;

public class GenState implements Encodable {

    static final int MAX_RELATIONS = 13;
    static final int MAX_ELEMENTS = 25;
    static final int DISPLACEMENT = MAX_RELATIONS * 2;

    List<Set<Byte>> modules;

    byte[] input;

    public double accReward = 0.0;

    Map<Element, Byte> indices;

    List<Element> modularizableElements;

    List<Relationship> relations;

    private GenState() {
        super();
    }

    public GenState(Map<Element, Byte> indices, List<Element> modularizableElements,
                    List<Relationship> relations) {
        super();
        this.indices = indices;
        this.modularizableElements = modularizableElements;
        this.relations = relations;
        input = new byte[DISPLACEMENT + MAX_ELEMENTS * (MAX_ELEMENTS - 1) / 2];
        // first we store relations in input layer
        for (int i = 0; i < relations.size(); i++) {
            Relationship r = relations.get(i);
            // for each relation we store the two entity indices it connects
            int indexA = i * 2;
            int indexB = i * 2 + 1;
            Entity entityA = r.a;
            Entity entityB = r.b;
            input[indexA] = indices.get(entityA);
            input[indexB] = indices.get(entityB);
        }
        // then we store for each pair of entities whether they are in the same module
        // or not
        // all the connections are zero at the beginning no need to call calculate
        // method
        modules = new ArrayList<>(modularizableElements.size());
        for (int i = 0; i < modularizableElements.size(); i++) {
            Set<Byte> nestedSet = new HashSet<>();
            Element element = modularizableElements.get(i);
            nestedSet.add(indices.get(element));
            modules.add(nestedSet);
        }
    }

    private void calculateConnections() {
        // then we store for each pair of entities whether they are in the same module
        // or not
        for (int i = 0; i < modularizableElements.size(); i++) {
            Element elementI = modularizableElements.get(i);
            Set<Byte> setI = findSet(i);
            byte indexI = (byte) (i + 1);
            for (int j = i + 1; j < modularizableElements.size(); j++) {
                Element elementJ = modularizableElements.get(j);
                Set<Byte> setJ = findSet(j);
                byte indexJ = (byte) (j + 1);
                input[DISPLACEMENT + getTriangularMatrixIndex(i, j)] = setI.contains(indexJ) || setJ.contains(indexI)
                        ? (byte) 1
                        : (byte) 0;
            }
        }
    }

    Pair<Set<Byte>, Set<Byte>> belongToSameModule(Element a, Element b) {
        Set<Byte> setI = null;
        Set<Byte> setJ = null;
        byte indexI = indices.get(a);
        byte indexJ = indices.get(b);
        for (Set<Byte> s : modules) {
            if (s.contains(indexI)) {
                setI = s;
            }
            if (s.contains(indexJ)) {
                setJ = s;
            }
        }
        return new Pair<>(setI, setJ);
    }

    private Set<Byte> findSet(int element) {
        byte elementIndex = (byte) (element + 1);
        for (Set<Byte> s : modules) {
            if (s.contains(elementIndex)) {
                return s;
            }
        }
        return null;
    }

    public void performAction(int action) {
        Pair<Integer, Integer> indices = getTriangularMatrixRowAndColumn(action);
        moveItoJ(indices.getFirst(), indices.getSecond());
    }

    private void moveItoJ(int i, int j) {
        if (i >= modularizableElements.size() || j >= modularizableElements.size()) {
            return;
        }
        Set<Byte> setI = null;
        Set<Byte> setJ = null;
        byte indexI = (byte) (i + 1);
        byte indexJ = (byte) (j + 1);
        for (Set<Byte> s : modules) {
            if (s.contains(indexI)) {
                setI = s;
            }
            if (s.contains(indexJ)) {
                setJ = s;
            }
        }
        if (setI != setJ) {
            //TODO
//            setI.remove(indexI);
//            setJ.add(indexI);
//            if (setI.isEmpty()) {
//                modules.remove(setI);
//            }
            //start
            setJ.addAll(setI);
            modules.remove(setI);
            //end
            calculateConnections();
        }
    }

    public List<Integer> getInvalidActions() {
        ArrayList<Integer> invalid = new ArrayList<>();
        for (Set<Byte> m : modules) {
            if (m.size() > 1) {
                Byte[] arr = m.toArray(new Byte[0]);
                for (int i = 0; i < arr.length; i++) {
                    for (int j = i + 1; j < arr.length; j++) {
                        invalid.add(getTriangularMatrixIndex(arr[i] - 1, arr[j] - 1));
                    }
                }
            }
        }
        return invalid;
    }

    Element getElementByIndex(byte index) {
        return modularizableElements.get(index - 1);
    }

    private int getTriangularMatrixIndex(int a, int b) {
        // https://stackoverflow.com/a/27682124
        int size = a * (a + 1) / 2;
        return size + b - 1;
    }

    private Pair<Integer, Integer> getTriangularMatrixRowAndColumn(int index) {
        // https://math.stackexchange.com/a/1417583
        double numerator = Math.sqrt(8 * index + 1) - 1;
        int row = (int) Math.floor(numerator / 2);
        int column = index + 1 - row * (row + 1) / 2;
        return new Pair<Integer, Integer>(row, column);
    }

    @Override
    public Encodable dup() {
        GenState newState = new GenState();
        newState.indices = indices;
        newState.modularizableElements = modularizableElements;
        newState.relations = relations;
        newState.accReward = accReward;
        newState.input = Arrays.copyOf(input, input.length);
        newState.modules = deepCopySet(modules);
        return newState;
    }

    private List<Set<Byte>> deepCopySet(List<Set<Byte>> set) {
        List<Set<Byte>> newSet = new ArrayList<>(set.size());
        for (Set<Byte> nestedSet : set) {
            Set<Byte> newNestedSet = new HashSet<>(nestedSet.size());
            newNestedSet.addAll(nestedSet);
            newSet.add(newNestedSet);
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
        for (Set<Byte> m : modules) {
            i++;
            System.out.print("Module " + i + ":");
            System.out.print(" [ ");
            for (byte b : m) {
                System.out.print(modularizableElements.get(b - 1).name + ", ");
            }
            System.out.println("]");
        }
        System.out.println("MQ Index = " + accReward);
    }

}
