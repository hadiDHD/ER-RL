package org.module.eer.exprelindi.exp;

import kotlin.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.module.eer.general.ModelGenerator;
import org.module.eer.model.ERModel;

public class ExpMDP implements MDP<ExpState, Integer, DiscreteSpace> {

    ExpState state;
    private ArrayObservationSpace<ExpState> observationSpace;
    private DiscreteSpace actionSpace;
    boolean isTraining;
    private int step;
    private ModelGenerator generator = new ModelGenerator();

    public ExpMDP() {
        isTraining = true;
        init(generator.generateModels());
    }

    public ExpMDP(ERModel er) {
        this.isTraining = false;
        init(er);
    }

    private void init(ERModel er) {
        state = new ExpState(er);
        observationSpace = new ArrayObservationSpace<>(new int[]{state.input.length});
        actionSpace = new DiscreteSpace(ExpState.OUTPUT_SIZE);
        step = 0;
    }

    @Override
    public ObservationSpace<ExpState> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public ExpState reset() {
        if (!isTraining) {
            state = new ExpState(state.er);
            step = 0;
            return state;
        }
        init(generator.generateModels());
        return state;
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<ExpState> step(Integer action) {
        Pair<Integer, Integer> p = ExpState.getTriangularMatrixRowAndColumn(action);
        double prevMq = state.mq;
        boolean valid = state.moveItoJ(p.getFirst(), p.getSecond());
        if (!valid) {
            return new StepReply<>(state, 0, false, null);
        }
        state.mq = ExpMQ.apply(state);
        step++;
        return new StepReply<>(state, state.mq - prevMq, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return state.modules.size() < 2 || state.mq > 6;
    }

    @Override
    public MDP<ExpState, Integer, DiscreteSpace> newInstance() {
        if (!isTraining) {
            return new ExpMDP(state.er);
        } else {
            return new ExpMDP();
        }
    }

}
