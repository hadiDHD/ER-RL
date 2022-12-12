package org.module.eer.exprel.exp;

import org.module.eer.model.Module;
import org.module.eer.model.Relationship;

public class ExpMQ {

    public static double apply(ExpState state) {
        for (Module m : state.modules) {
            m.getMQ().reset();
        }
        for (Relationship r : state.er.relationships) {
            Module moduleR = null;
            Module moduleA = null;
            Module moduleB = null;
            for (Module m : state.modules) {
                if (m.contains(r)) {
                    moduleR = m;
                }
                if (m.contains(r.a)) {
                    moduleA = m;
                }
                if (m.contains(r.b)) {
                    moduleB = m;
                }
            }
            if (moduleA == moduleR) {
                moduleA.getMQ().intraLink++;
            } else {
                moduleA.getMQ().interLink++;
                moduleR.getMQ().interLink++;
            }
            if (moduleB == moduleR) {
                moduleB.getMQ().intraLink++;
            } else {
                moduleB.getMQ().interLink++;
                moduleR.getMQ().interLink++;
            }
        }
//        double sum = 0;
//        for (Module m : state.modules) {
//            sum += m.getMQ().getMQ();
//        }
        return state.getMQ();
    }
}