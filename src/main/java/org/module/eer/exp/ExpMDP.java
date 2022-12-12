package org.module.eer.exp;

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
        int size = ExpState.MAX_ENTITIES;
        actionSpace = new DiscreteSpace(size * (size - 1) / 2);
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
        double prevReward = state.getMQ();
        state.performAction(action);
        double reward = ExpMQ.apply(state);
        step++;
        return new StepReply<>(state, reward - prevReward, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return state.modules.size() < 2 || step >= ExpTrainer.epochStep;
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
