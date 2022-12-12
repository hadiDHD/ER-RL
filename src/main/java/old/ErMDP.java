package old;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;

import java.util.Arrays;

public class ErMDP implements MDP<ErState, Integer, DiscreteSpace> {

    int size;
    private ArrayObservationSpace<ErState> observationSpace;
    private DiscreteSpace actionSpace;
    private boolean isTraining;
    private int step = 0;
    double bestReward = 0.0;
    byte[] bestData = null;

    public ErMDP(int size, boolean isTraining) {
        this.size = size;
        state = new ErState(size);
        observationSpace = new ArrayObservationSpace<>(
                new int[]{size * 2 - 1}
        );
        actionSpace = new DiscreteSpace(size * size + size - 1);
        this.isTraining = isTraining;
    }

    ErState state;

    @Override
    public ObservationSpace<ErState> getObservationSpace() {
        return observationSpace;
    }

    @Override
    public DiscreteSpace getActionSpace() {
        return actionSpace;
    }

    @Override
    public ErState reset() {
        state = new ErState(size);
        step = 0;
        bestData = null;
        bestReward = 0.0;
        return state;
    }

    @Override
    public void close() {

    }

    @Override
    public StepReply<ErState> step(Integer action) {
        if (action < size * size) {
            // swap elements i and j
            int i = action / size * 2;
            int j = action % size * 2;
            state.swap(i, j);
        } else {
            // toggle separator i
            int i = (action - size * size) * 2 + 1;
            state.toggle(i);
        }
        // TODO: calculate new reward
        double reward = frequency(state.data, (byte) 1);
        if (!isTraining) {
            if (reward > bestReward) {
                bestData = Arrays.copyOf(state.data, state.data.length);
                bestReward = reward;
            }
        }
        step++;
        return new StepReply<>(state, reward, isDone(), null);
    }

    @Override
    public boolean isDone() {
        return !isTraining && step >= Trainer.epochStep;
    }

    @Override
    public MDP<ErState, Integer, DiscreteSpace> newInstance() {
        return new ErMDP(size, isTraining);
    }

    public static int frequency(byte[] c, byte o) {
        int result = 0;
        for (byte e : c)
            if (o == e)
                result++;
        return result;
    }
}
