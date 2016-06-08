"""
Analysis/simulation context classes for scanr.
"""

from tenko.factory import ContextFactory

from . import SRCDIR, PROJECT_ROOT


ScanrAnalysis = ContextFactory.analysis_class("ScanrAnalysis",
    SRCDIR, PROJECT_ROOT, logcolor='purple')

ScanrSimulation = ContextFactory.simulation_class("ScanrSimulation",
    SRCDIR, PROJECT_ROOT, logcolor='orange')
