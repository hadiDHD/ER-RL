package org.module.eer.exp;

import com.google.gson.Gson;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.module.eer.general.GenMDP;
import org.module.eer.general.GenSolver;
import org.module.eer.general.GenState;
import org.module.eer.model.ERModel;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ExpTrainerStaged {

    static int epochStep = 1000;
    static int epochNumber = 1;

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(2)
                    .numHiddenNodes(ExpState.DISPLACEMENT).
                    build();

    public static void main(String[] args) {
        try {
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
//            DQNPolicy<ExpState> policy = null;
            DQNPolicy<ExpState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\exp\\1649675732768");
            QLearningDiscreteDense<ExpState> qLearning;
            for (int i = 0; i < 100; i++) {
                QLearningConfiguration QL =
                        QLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(1000)
                                .gamma(0.9)
                                .rewardFactor(0.5)
                                .build();
                if (policy == null) {
                    qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
                } else {
                    qLearning = new QLearningDiscreteDense<>(mdp, policy.getNeuralNet(), QL);
                }
                qLearning.train();
                policy = qLearning.getPolicy();

                int validActionRange = model.entities.size() * (model.entities.size() - 1) / 2;
                double reward = ExpTrainer.play(qLearning.getPolicy(), mdp, validActionRange);

                if (i % 10 == 9) {
                    try {
                        File nn = new File("neural networks/exp/" + System.currentTimeMillis());
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

