from .data import PatID, Task, Stim


missing_datasets = {
    PatID.ET3: [Task.Pouring],
    #PatID.ET4: [Task.Pouring], # Pouring ON saturates
    PatID.ET5: [Task.Pegboard],
    PatID.ET6: [Task.Pegboard],
    PatID.ET8: [Task.Pegboard],
}

consolidate_chan_dict = {
    PatID.ET1: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: [1]},
        Task.Pouring: {Stim.ON: None, Stim.OFF: [1]},
        Task.Posture: {Stim.ON: None, Stim.OFF: [1]},
    },
    PatID.ET2: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: None, Stim.OFF: None},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
    PatID.ET3: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: None, Stim.OFF: None},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
    PatID.ET4: {
        Task.Pegboard: {Stim.ON: [2, 5], Stim.OFF: [2, 5]},
        Task.Pouring: {Stim.ON: [2, 5], Stim.OFF: [2, 5]},
        Task.Posture: {Stim.ON: [2, 5], Stim.OFF: [2, 5]},
    },
    PatID.ET5: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: [2, 5], Stim.OFF: [2, 5]},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
    PatID.ET6: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: [2, 5], Stim.OFF: [2, 5]},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
    PatID.ET7: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: None, Stim.OFF: None},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
    PatID.ET8: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: None, Stim.OFF: None},
        Task.Posture: {Stim.ON: None, Stim.OFF: None},
    },
}

raw_data_fnames = {
    PatID.ET1: {
        Task.Pegboard: {Stim.ON: "Pegboard_on", Stim.OFF: "Pegboard_off"},
        Task.Pouring: {Stim.ON: "Pouring_on", Stim.OFF: "Pouring_off"},
        Task.Posture: {Stim.ON: "Posture_on", Stim.OFF: "Posture_off"},
    },
    PatID.ET2: {
        Task.Pegboard: {Stim.ON: "Pegboard_on", Stim.OFF: "Pegboard_off"},
        Task.Pouring: {Stim.ON: "Pouring_on", Stim.OFF: "Pouring_off"},
        Task.Posture: {Stim.ON: "Posture_on", Stim.OFF: "Posture_off"},
    },
    PatID.ET3: {
        Task.Pegboard: {
            Stim.ON: "ET_1Jun_Pegboard_BiStimLR0",
            Stim.OFF: "ET_1Jun_Pegboard_NoStim_BiLFP",
        },
        Task.Pouring: {Stim.ON: None, Stim.OFF: None},
        Task.Posture: {
            Stim.ON: "ET_1Jun_Posture_BiStimLR0",
            Stim.OFF: "ET_1Jun_Posture_NoStim_BiLFP",
        },
    },
    PatID.ET4: {
        Task.Pegboard: {
            Stim.ON: "ET_2Jun_Pegboard_BiStimLR3",
            Stim.OFF: "ET_2Jun_Pegboard_BiLFP_NoStim",
        },
        Task.Pouring: {
            Stim.ON: "ET2_02jUN_pOURING",
            Stim.OFF: "ET2_02Jun_Pouring_Nostim",
        },
        Task.Posture: {
            Stim.ON: "ET_2Jun_Posture_BiStimLR3",
            Stim.OFF: "ET_2Jun_Posture_BiLFP_NoStim",
        },
    },
    PatID.ET5: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {
            Stim.ON: "ET_5thOct_Pouring_Stim",
            Stim.OFF: "ET_5thOct_Pouring_NoStim",
        },
        Task.Posture: {
            Stim.ON: "ET_5thOct_PostureHolding_Stim",
            Stim.OFF: "ET_5thOct_PostureHolding_NoStim",
        },
    },
    PatID.ET6: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {
            Stim.ON: "ET6_5Oct_Pouring_Stim",
            Stim.OFF: "ET6_5Oct_Pouring_NoStim",
        },
        Task.Posture: {
            Stim.ON: "ET6_5Oct_PostureHolding_Stim",
            Stim.OFF: "ET6_5Oct_PostureHolding_NoStim",
        },
    },
    PatID.ET7: {
        Task.Pegboard: {Stim.ON: "Pegboard_On", Stim.OFF: "Pegboard_off"},
        Task.Pouring: {Stim.ON: "Pouring_On", Stim.OFF: "Pouring_Off"},
        Task.Posture: {Stim.ON: "Posture_On", Stim.OFF: "Posture_off"},
    },
    PatID.ET8: {
        Task.Pegboard: {Stim.ON: None, Stim.OFF: None},
        Task.Pouring: {Stim.ON: "Ox08Dec2019_Pouring_on", Stim.OFF: "Pouring_off"},
        Task.Posture: {
            Stim.ON: "OxMS08Dec_Posture_StimOn",
            Stim.OFF: "OxMS08Dec_Posture_StimOff",
        },
    },
}
