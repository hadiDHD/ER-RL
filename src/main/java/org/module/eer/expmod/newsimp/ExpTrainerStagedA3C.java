package org.module.eer.expmod.newsimp;

import com.google.gson.Gson;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.module.eer.model.ERModel;
import org.nd4j.linalg.learning.config.Adam;

import javax.swing.*;
import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;

public class ExpTrainerStagedA3C {

    static int epochStep = ExpState.MAX_ENTITIES;

    private static final ActorCriticDenseNetworkConfiguration NET_A3C = ActorCriticDenseNetworkConfiguration.builder()
            .numLayers(3)
            .numHiddenNodes(ExpState.OUTPUT_SIZE)
            .learningRate(0.01)
            .updater(new Adam(0.001))
            .l2(0.000)
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
            System.out.println(model.entities.size());

            // close reader
            reader.close();
//            ExpMDP mdpIndi = new ExpMDP(model);
            ExpMDP mdp = new ExpMDP();
            MyPolicy policy = null;
//            String loadName = "1650553323713";
//            MyPolicy policy = MyPolicy.load(
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\modsimp\\" + loadName + "_value",
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\modsimp\\" + loadName + "_policy"
//            );
            MyA3CDiscreteDense dql;
            for (int i = 0; i < 1; i++) {
                A3CLearningConfiguration LEARNING_A3C =
                        A3CLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(1000000)
                                .numThreads(64)
                                .rewardFactor(0.01)
                                .gamma(0.99)
                                .nStep(5)
                                .build();
                if (policy == null) {
                    dql = new MyA3CDiscreteDense(mdp, NET_A3C, LEARNING_A3C);
                } else {
                    dql = new MyA3CDiscreteDense(mdp, policy.getNeuralNet(), LEARNING_A3C);
                }
                dql.addListener(new TrainingListener() {
                    @Override
                    public ListenerResponse onTrainingStart() {
//                        System.out.println("onTrainingStart");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public void onTrainingEnd() {
//                        System.out.println("onTrainingStart");
                    }

                    @Override
                    public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
//                        System.out.println("onTrainingStart");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, IDataManager.StatEntry statEntry) {
//                        System.out.println("onEpochTrainingResult");
                        return ListenerResponse.CONTINUE;
                    }

                    @Override
                    public ListenerResponse onTrainingProgress(ILearning learning) {
                        System.out.println("onTrainingProgress");
                        System.out.println("stepCount:" + learning.getStepCount());
                        MyPolicy policy = ((MyPolicy) learning.getPolicy()).copy();
                        new Thread(() -> {
                            ExpMDP playMdp = new ExpMDP(model);
                            ExpState bestState = policy.play(playMdp);
                            for (int p = 0; p < 1000; p++) {
                                playMdp = new ExpMDP(model);
                                ExpState newState = policy.play(playMdp);
                                if (newState.reward > bestState.reward) {
                                    bestState = newState;
                                }
                            }
                            bestState.print();
                            if (true) {
                                try {
                                    String name = System.currentTimeMillis() + "";
                                    File valueN = new File("neural networks/modsimp/" + name + "_value");
                                    File policyN = new File("neural networks/modsimp/" + name + "_policy");
                                    System.out.println(valueN.getAbsolutePath());
//                        nn.createNewFile();
                                    policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }
                        }).start();
                        return ListenerResponse.CONTINUE;
                    }
                });
                dql.train();

                ExpMDP playMdp = new ExpMDP(model);
                policy = (MyPolicy) dql.getPolicy();

//                int validActionRange = ExpState.OUTPUT_SIZE;
                ExpState bestState = policy.play(playMdp);
                for (int p = 0; p < 1000; p++) {
                    playMdp = new ExpMDP(model);
                    ExpState newState = policy.play(playMdp);
                    if (newState.reward > bestState.reward) {
                        bestState = newState;
                    }
                }
                bestState.print();

                if (true) {
                    try {
                        String name = System.currentTimeMillis() + "";
                        File valueN = new File("neural networks/modsimp/" + name + "_value");
                        File policyN = new File("neural networks/modsimp/" + name + "_policy");
                        System.out.println(valueN.getAbsolutePath());
//                        nn.createNewFile();
                        policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
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

