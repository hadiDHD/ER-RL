package org.module.eer.expmod.simp;

import kotlin.Pair;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.module.eer.general.ModelGenerator;
import org.module.eer.model.ERModel;
import org.module.eer.model.Module;

public class ExpMDP implements MDP<ExpState, Integer, DiscreteSpace> {

    public ExpState state;
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
        Pair<Integer, Integer> p = org.module.eer.exp.ExpState.getTriangularMatrixRowAndColumn(action);
        double prevMq = state.modules.get(p.getFirst()).getMQ().getMQ() + state.modules.get(p.getSecond()).getMQ().getMQ();
        Module mJ = state.moveItoJ(p.getFirst(), p.getSecond());
        state.reward = ExpMQ.apply(state);
        step++;
        return new StepReply<>(state, mJ.getMQ().getMQ() - prevMq, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return state.modules.size() <= 2 || step >= ExpTrainerStagedA3C.epochStep;
    }

    @Override
    public MDP<ExpState, Integer, DiscreteSpace> newInstance() {
        if (!isTraining) {
            return new ExpMDP(state.er);
        } else {
            return new ExpMDP();
        }
    }

    public Integer getValidActionRange() {
        return state.modules.size() * (state.modules.size() - 1) / 2;
    }

}
