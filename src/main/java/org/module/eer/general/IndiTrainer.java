package org.module.eer.general;

import com.google.gson.Gson;
import org.deeplearning4j.rl4j.learning.configuration.QLearningConfiguration;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.configuration.DQNDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.module.eer.model.ERModel;
import org.nd4j.linalg.learning.config.Adam;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;

public class IndiTrainer {

    static int epochStep = Integer.MAX_VALUE;
    static int epochNumber = 1;

    static QLearningConfiguration QL =
            QLearningConfiguration.builder()
                    .maxEpochStep(epochStep)
                    .maxStep(100000)
                    .gamma(0)
                    .rewardFactor(1)
                    .build();

    static DQNDenseNetworkConfiguration NET =
            DQNDenseNetworkConfiguration.builder()
                    .numLayers(3)
                    .numHiddenNodes(25).
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
            //TODO
//            GenMDP mdp = new GenMDP(model.indices, model.elements, model.relationships);
            GenMDP mdp = new GenMDP(Collections.emptyMap(), model.elements, model.relationships);
            //TODO
            DQNPolicy<GenState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\staged\\1649241171063");
            QLearningDiscreteDense<GenState> qLearning = new QLearningDiscreteDense<>(mdp, policy.getNeuralNet(), QL);
//            QLearningDiscreteDense<GenState> qLearning = new QLearningDiscreteDense<>(mdp, NET, QL);
            qLearning.train();
            File nn = new File("neural networks/" + System.currentTimeMillis());
            System.out.println(nn.getAbsolutePath());
            try {
                nn.createNewFile();
                qLearning.getPolicy().save(nn.getAbsolutePath());
            } catch (IOException e) {
                e.printStackTrace();
            }

            int validActionRange = model.elements.size() * (model.elements.size() - 1) / 2;
            GenSolver.play(qLearning.getPolicy(), mdp, validActionRange);

        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}

