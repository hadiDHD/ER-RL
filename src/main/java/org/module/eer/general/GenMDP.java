package org.module.eer.general;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.module.eer.model.Element;
import org.module.eer.model.Relationship;

public class GenMDP implements MDP<GenState, Integer, DiscreteSpace> {

    GenState state;
    private ArrayObservationSpace<GenState> observationSpace;
    private DiscreteSpace actionSpace;
    boolean isTraining;
    private int step;

    public GenMDP() {
        isTraining = true;
        ModelGenerator generator = new ModelGenerator();
        generator.generateModels();
        init(generator.indices, generator.elements, generator.relationships);
    }

    public GenMDP(Map<Element, Byte> indices, List<Element> modularizableElements,
                  List<Relationship> relations) {
        this.isTraining = false;
        init(indices, modularizableElements, relations);
    }

    private void init(Map<Element, Byte> indices, List<Element> modularizableElements,
                      List<Relationship> relations) {
        state = new GenState(indices, modularizableElements, relations);
        observationSpace = new ArrayObservationSpace<>(new int[]{state.input.length});
        int size = GenState.MAX_ELEMENTS;
        actionSpace = new DiscreteSpace(size * (size - 1) / 2);
        step = 0;
    }

    @Override
    public ObservationSpace<GenState> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public GenState reset() {
        if (!isTraining) {
            state = new GenState(state.indices, state.modularizableElements, state.relations);
            step = 0;
            return state;
        }
        ModelGenerator generator = new ModelGenerator();
        generator.generateModels();
        init(generator.indices, generator.elements, generator.relationships);
        return state;
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<GenState> step(Integer action) {
        double accReward = state.accReward;
        state.performAction(action);
        double reward = GenMQ.apply(state);
        step++;
        state.accReward = reward;
        return new StepReply<>(state, reward - accReward, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return state.modules.size() < 2 || step >= GenTrainer.epochStep;
    }

    @Override
    public MDP<GenState, Integer, DiscreteSpace> newInstance() {
        if (!isTraining) {
            return new GenMDP(state.indices, state.modularizableElements, state.relations);
        }else {
            return new GenMDP();
        }
    }

}
