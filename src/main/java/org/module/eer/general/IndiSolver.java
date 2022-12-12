package org.module.eer.general;

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
import java.util.Collections;

public class IndiSolver {

    public static void main(String[] args) {
        try {
            DQNPolicy<GenState> policy = DQNPolicy.load("C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\1649160388154");
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
//            GenMDP mdp = new GenMDP(model.indices, model.elements, model.relationships);\
            GenMDP mdp = new GenMDP(Collections.emptyMap(), model.elements, model.relationships);


            int validActionRange = model.elements.size() * (model.elements.size() - 1) / 2;
            GenSolver.play(policy, mdp, validActionRange);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}

