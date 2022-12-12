package org.module.eer.expmod.simp;

import com.google.gson.Gson;
import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.configuration.A3CLearningConfiguration;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.network.configuration.ActorCriticDenseNetworkConfiguration;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.rl4j.policy.ACPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.IDataManager;
import org.deeplearning4j.rl4j.util.LegacyMDPWrapper;
import org.module.eer.general.GenSolver;
import org.module.eer.model.ERModel;
import org.nd4j.linalg.api.ndarray.INDArray;
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
            ExpMDP mdp = new ExpMDP(model);
            MyPolicy policy = null;
//            String loadName = "1650553323713";
//            ACPolicy<ExpState> policy = ACPolicy.load(
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\mod\\" + loadName + "_value",
//                    "C:\\Users\\hadid\\IdeaProjects\\er\\neural networks\\mod\\" + loadName + "_policy"
//            );
            MyA3CDiscreteDense dql;
            for (int i = 0; i < 1; i++) {
                A3CLearningConfiguration LEARNING_A3C =
                        A3CLearningConfiguration.builder()
                                .maxEpochStep(epochStep)
                                .maxStep(1000000)
                                .numThreads(16)
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
                        ACPolicy policy = (ACPolicy) learning.getPolicy();
                        double reward = play(policy, mdp);
                        if (false) {
                            try {
                                String name = System.currentTimeMillis() + "";
                                File valueN = new File("neural networks/mod/" + name + "_value");
                                File policyN = new File("neural networks/mod/" + name + "_policy");
                                System.out.println(valueN.getAbsolutePath());
//                        nn.createNewFile();
                                policy.save(valueN.getAbsolutePath(), policyN.getAbsolutePath());
                            } catch (IOException e) {
                                e.printStackTrace();
                            }
                        }
                        return ListenerResponse.CONTINUE;
                    }
                });
                dql.train();
                policy = (MyPolicy) dql.getPolicy();

//                int validActionRange = ExpState.OUTPUT_SIZE;
                double reward = play((ACPolicy) dql.getPolicy(), mdp);

                if (false) {
                    try {
                        String name = System.currentTimeMillis() + "";
                        File valueN = new File("neural networks/mod/" + name + "_value");
                        File policyN = new File("neural networks/mod/" + name + "_policy");
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

    public static double play(ACPolicy policy, ExpMDP mdp) {
        LegacyMDPWrapper<ExpState, Integer, DiscreteSpace> mdpWrapper = new LegacyMDPWrapper<>(mdp, null);

        Learning.InitMdp<Observation> initMdp = GenSolver.refacInitMdp(mdpWrapper, null);
        Observation obs = initMdp.getLastObs();

        double reward = initMdp.getReward();

        int lastAction = mdpWrapper.getActionSpace().noOp();
        int action;

        ExpState bestState = (ExpState) mdp.state.dup();
        int goodActions = 0;

        while (!mdpWrapper.isDone()) {

            INDArray output = policy.getNeuralNet().outputAll(obs.getData())[1];
            double[] actions = output.toDoubleVector();
//            List<Integer> invalidActions = mdp.state.getInvalidActions();
//            for (int ac : invalidActions) {
//                actions[ac] = 0.0;
//            }
            int moduleSize = mdp.state.modules.size();
            int actionRange = moduleSize * (moduleSize - 1) / 2;
            double maxActionValue = Double.MIN_VALUE;
            int maxActionIndex = -1;
            for (int i = 0; i < actionRange; i++) {
                if (actions[i] > maxActionValue) {
                    maxActionValue = actions[i];
                    maxActionIndex = i;
                }
            }
            action = maxActionIndex;
            if (maxActionIndex == -1) {
                break;
            }

//            System.out.println("actions: " + Arrays.toString(actions));
//            System.out.println("invalids: " + Arrays.toString(invalidActions.toArray()));
//            Pair<Integer, Integer> p = ExpState.getTriangularMatrixRowAndColumn(maxActionIndex);
//            Element eI = mdp.state.er.elements.get(p.getFirst());
//            Element eJ = mdp.state.er.elements.get(p.getSecond());
//            System.out.println("move: " + eI + " to " + eJ);

            StepReply<Observation> stepReply = mdpWrapper.step(action);
            reward += stepReply.getReward();

//            System.out.println("Reward: " + reward);

            obs = stepReply.getObservation();

            if (mdp.state.reward > bestState.reward) {
                bestState = (ExpState) mdp.state.dup();
//                System.out.println("YEEEEEEEEEEEEEEEEEEES!!!!");
                goodActions++;
            }
//            if(mdp.state.modules.size() < 8){
//                mdp.state.print();
//            }
        }

//        System.out.println(ExpMQ.apply(bestState));
        bestState.print();
        System.out.println("good actions: " + goodActions);
        return reward;
    }
}

