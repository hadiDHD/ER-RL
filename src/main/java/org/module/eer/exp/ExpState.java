package org.module.eer.exp;

import kotlin.Pair;
import org.deeplearning4j.rl4j.space.Encodable;
import org.module.eer.model.ERModel;
import org.module.eer.model.Element;
import org.module.eer.model.Entity;
import org.module.eer.model.Module;
import org.module.eer.model.Relationship;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

public class ExpState implements Encodable {

    static final int MAX_ENTITIES = 12;
    static final int DISPLACEMENT = MAX_ENTITIES * (MAX_ENTITIES - 1) / 2;

    Set<Module> modules;

    byte[] input;

    ERModel er;

    private ExpState() {
        super();
    }

    public ExpState(ERModel er) {
        super();
        this.er = er;
        input = new byte[DISPLACEMENT * 2];

        modules = new HashSet<>(er.entities.size());
        for (int i = 0; i < er.entities.size(); i++) {
            Module m = new Module();
            Element element = er.entities.get(i);
            m.addElement(element);
            modules.add(m);
        }
        // first we store relations in input layer
        for (int i = 0; i < er.relationships.size(); i++) {
            Relationship r = er.relationships.get(i);
            // for each relation we store the two entity indices it connects
            Entity entityA = r.a;
            Entity entityB = r.b;
            int indexA = er.indices.get(entityA);
            int indexB = er.indices.get(entityB);
            int index = getTriangularMatrixIndex(indexA, indexB);
            input[index]++;
            findModule(Math.random() < 0.5 ? r.a : r.b).addElement(r);
        }
        // then we store for each pair of entities whether they are in the same module
        // or not
        // all the connections are zero at the beginning no need to call calculate
        // method
    }

    private void calculateConnections() {
        // then we store for each pair of entities whether they are in the same module
        // or not
        for (int i = 0; i < er.entities.size(); i++) {
            Module moduleI = findModule(er.entities.get(i));
            for (int j = i + 1; j < er.entities.size(); j++) {
                Module moduleJ = findModule(er.entities.get(j));
                input[DISPLACEMENT + getTriangularMatrixIndex(j, i)] = moduleI == moduleJ
                        ? (byte) 1
                        : (byte) 0;
            }
        }
    }

    private Module findModule(Element element) {
        for (Module m : modules) {
            if (m.contains(element)) {
                return m;
            }
        }
        throw new RuntimeException("Module not found!");
    }

    public void performAction(int action) {
        Pair<Integer, Integer> indices = getTriangularMatrixRowAndColumn(action);
        moveItoJ(indices.getFirst(), indices.getSecond());
    }

    private void moveItoJ(int i, int j) {
        if (i >= er.entities.size() || j >= er.entities.size()) {
            throw new RuntimeException("WRONG!");
        }
        Module moduleI = null;
        Module moduleJ = null;
        Entity eI = er.entities.get(i);
        Entity eJ = er.entities.get(j);
        for (Module m : modules) {
            if (m.contains(eI)) {
                moduleI = m;
            }
            if (m.contains(eJ)) {
                moduleJ = m;
            }
        }
        if (moduleI == null || moduleJ == null) {
            throw new RuntimeException("Could not find the module!");
        }
        if (moduleI != moduleJ) {
            //TODO
//            setI.remove(indexI);
//            setJ.add(indexI);
//            if (setI.isEmpty()) {
//                modules.remove(setI);
//            }
            //start
            moduleJ.merge(moduleI);
            modules.remove(moduleI);
            //end
            calculateConnections();
        }
    }

    public List<Integer> getInvalidActions() {
        ArrayList<Integer> invalid = new ArrayList<>();
        for (Module m : modules) {
            if (m.getEntities().size() > 1) {
                Entity[] entities = m.getEntities().toArray(new Entity[0]);
                for (int i = 0; i < entities.length; i++) {
                    for (int j = i + 1; j < entities.length; j++) {
                        invalid.add(getTriangularMatrixIndex(er.indices.get(entities[i]), er.indices.get(entities[j])));
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
        return newState;
    }

    private Set<Module> deepCopyModules() {
        Set<Module> newSet = new HashSet<>(modules.size());
        for (Module m : modules) {
            newSet.add(m.copy());
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
            i++;
            System.out.println(m.toString(i));
//            System.out.print("Module " + i + ":");
//            System.out.print(" [ ");
//            for (byte b : m) {
//                System.out.print(er.elements.get(b).name + ", ");
//            }
//            System.out.println("]");
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
