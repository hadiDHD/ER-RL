package org.module.eer.exprelindi.exp;

import com.google.gson.Gson;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.module.eer.model.ERModel;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ExpTrainerStaged {

    static int epochStep = ExpState.MAX_ELEMENTS;
    static int epochNumber = 1;

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(3)
                    .numHiddenNodes(ExpState.OUTPUT_SIZE)
                    .learningRate(0.001)
                    .build();

    public static void main(String[] args) {
        try {
            System.setProperty("org.bytedeco.openblas.load", "mkl");
            // create Gson instance
            Gson gson = new Gson();

            JFileChooser chooser = new JFileChooser();

            chooser.showOpenDialog(null);

            // create a reader
            Reader reader = Files.newBufferedReader(Paths.get(chooser.getSelectedFile().getAbsolutePath()));

            // convert JSON string to User object
            ERModel model = gson.fromJson(reader, ERModel.class);
            model.init();

            // close reader
            reader.close();
            ExpMDP mdp = new ExpMDP(model);
            DQNPolicy<ExpState> policy = null;
//            DQNPolicy<ExpState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\exp\\all\\1649767932798");
            QLearningDiscreteDense<ExpState> qLearning;
            for (int i = 0; i < 10; i++) {
                QLearningConfiguration QL =
                        QLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(10000)
                                .gamma(0.99)
                                .rewardFactor(0.01)
                                .updateStart(500)
                                .epsilonNbStep(1000)
                                .errorClamp(1.0)
                                .minEpsilon(0.1)
                                .doubleDQN(true)
                                .build();
                if (policy == null) {
                    qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
                } else {
                    qLearning = new QLearningDiscreteDense<>(mdp, policy.getNeuralNet(), QL);
                }
                qLearning.train();
                policy = qLearning.getPolicy();

                int validActionRange = ExpState.OUTPUT_SIZE;
                double reward = ExpTrainer.play(qLearning.getPolicy(), mdp, validActionRange);

                if (true) {
                    try {
                        File nn = new File("neural networks/exp/all/" + System.currentTimeMillis());
                        System.out.println(nn.getAbsolutePath());
                        nn.createNewFile();
                        policy.save(nn.getAbsolutePath());
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}

