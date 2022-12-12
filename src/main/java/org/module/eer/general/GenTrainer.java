package org.module.eer.general;

import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;

import java.io.File;
import java.io.IOException;

public class GenTrainer {

    static int epochStep = 100;
    static int epochNumber = 100;

    static QLearningConfiguration QL =
            QLearningConfiguration.builder()
                    .maxEpochStep(epochStep)
                    .maxStep(epochStep * epochNumber)
                    .gamma(0.95)
                    .rewardFactor(0.05)
                    .build();

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(3)
                    .numHiddenNodes(10).
                    build();

    public static void main(String[] args) {
        GenMDP mdp = new GenMDP();
        QLearningDiscreteDense<GenState> qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
        qLearning.train();
        File nn = new File("neural networks/" + System.currentTimeMillis());
        System.out.println(nn.getAbsolutePath());
        try {
            nn.createNewFile();
            qLearning.getPolicy().save(nn.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

