package old;

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;

import java.util.Arrays;

public class Trainer {

    static int epochStep = 1000;
    static int epochNumber = 100;

    static QLearningConfiguration QL =
            QLearningConfiguration.builder()
                    .maxEpochStep(epochStep)
                    .maxStep(epochStep * epochNumber)
                    .gamma(0.1)
                    .rewardFactor(0.9)
                    .build();

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(3)
                    .numHiddenNodes(20).
                    build();

    public static void main(String[] args) {
        ErMDP mdp = new ErMDP(10, true);
        QLearningDiscreteDense<ErState> qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
        qLearning.train();
        mdp = new ErMDP(10, false);
        qLearning.getPolicy().play(mdp);
        System.out.println(Arrays.toString(mdp.bestData));
        System.out.println("Reward: " + mdp.bestReward);
    }
}
