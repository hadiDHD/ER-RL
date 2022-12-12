package org.module.eer.expmod.newsimp;

import org.module.eer.model.Module;
import org.module.eer.model.Relationship;

public class ExpMQ {

    public static double apply(ExpState state) {
        for (Module m : state.modules) {
            m.getMQ().reset();
        }
        state.resetInput();
        for (Relationship r : state.er.relationships) {
            int moduleR = -1;
            int moduleA = -1;
            int moduleB = -1;
            for (int i = 0; i < state.modules.size(); i++) {
                Module m = state.modules.get(i);
                if (m.contains(r)) {
                    moduleR = i;
                }
                if (m.contains(r.a)) {
                    moduleA = i;
                }
                if (m.contains(r.b)) {
                    moduleB = i;
                }
            }
            if (moduleA == moduleR) {
                state.modules.get(moduleA).getMQ().intraLink++;
                state.increaseInput(moduleA);
            } else {
                state.modules.get(moduleA).getMQ().interLink++;
                state.modules.get(moduleR).getMQ().interLink++;
                state.decreaseInput(moduleA, moduleR);
            }
            if (moduleB == moduleR) {
                state.modules.get(moduleB).getMQ().intraLink++;
                state.increaseInput(moduleB);
            } else {
                state.modules.get(moduleB).getMQ().interLink++;
                state.modules.get(moduleR).getMQ().interLink++;
                state.decreaseInput(moduleB, moduleR);
            }
        }
        return state.getMQ();
    }
}