#!/usr/bin/env python3

# ******************************************
#
#    SHARC Program Suite
#
#    Copyright (c) 2025 University of Vienna
#
#    This file is part of SHARC.
#
#    SHARC is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    SHARC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    inside the SHARC manual.  If not, see <http://www.gnu.org/licenses/>.
#
# ******************************************


import datetime
import importlib.metadata
import os
import shutil
from io import TextIOWrapper

import numpy as np
from SHARC_FAST import SHARC_FAST
from sgdml.predict import GDMLPredict
from utils import expand_path, link, question

__all__ = ["SHARC_SGDML"]

AUTHORS = "Simon Horacek"
VERSION = "0.1.0"
VERSIONDATE = datetime.datetime(2026, 3, 10)
NAME = "SGDML"
DESCRIPTION = "     FAST interface for sGDML"

CHANGELOGSTRING = """
"""


all_features = set(
    [
        "h",
        "grad",
    ]
)

class SHARC_SGDML(SHARC_FAST):
    """
    SHARC interface for sGDML
    """
    _version = VERSION
    _versiondate = VERSIONDATE
    _authors = AUTHORS
    _changelogstring = CHANGELOGSTRING
    _name = NAME
    _description = DESCRIPTION

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Add resource keys
        self.QMin.resources.update({"modelpath": None})
        self.QMin.resources.types.update({"modelpath": str})

        # Add template keys
        self.QMin.template.update({"cutoff": 10.0, "nac_key": "smooth_nacs", "properties": ["energy", "forces"]})
        self.QMin.template.types.update({"cutoff": float, "nac_key": str, "properties": list})

        self.GDMLpredict = None
        self._resources_file = None
        self._template_file = None
    
    @staticmethod
    def version() -> str:
        return SHARC_SGDML._version

    @staticmethod
    def versiondate() -> datetime.datetime:
        return SHARC_SGDML._versiondate

    @staticmethod
    def changelogstring() -> str:
        return SHARC_SGDML._changelogstring

    @staticmethod
    def authors() -> str:
        return SHARC_SGDML._authors
    
    @staticmethod
    def name() -> str:
        return SHARC_SGDML._name

    @staticmethod
    def description() -> str:
        return SHARC_SGDML._description
    
    @staticmethod
    def about() -> str:
        return f"{SHARC_SGDML._name}\n{SHARC_SGDML._description}"

    def get_features(self, KEYSTROKES: TextIOWrapper | None = None) -> set[str]:
        """return available features

        ---
        Parameters:
        KEYSTROKES: object as returned by open() to be used with question()
        """
        return all_features
    
    def get_infos(self, INFOS: dict, KEYSTROKES: TextIOWrapper | None = None) -> dict:
        self.log.info("=" * 80)
        self.log.info(f"{'||':<78}||")
        self.log.info(f"||{'SGDML interface setup': ^76}||\n{'||':<78}||")
        self.log.info("=" * 80)
        self.log.info("\n")

        # Template file handling
        if os.path.isfile("SGDML.template"):
            self.log.info("Found SGDML.template in current directory")
            if question("Use this template file?", bool, KEYSTROKES=KEYSTROKES, default=True):
                self._template_file = "SGDML.template"
        else:
            self.log.info("Specify a path to a SGDML template file.")
            while not os.path.isfile(template_file := question("Template path:", str, KEYSTROKES=KEYSTROKES)):
                self.log.info(f"File {template_file} does not exist!")
            self._template_file = template_file

        # Resource file handling
        if question("Do you have a SGDML.resources file?", bool, KEYSTROKES=KEYSTROKES, autocomplete=False, default=False):
            while not os.path.isfile(
                resources_file := question("Specify path to SGDML.resources", str, KEYSTROKES=KEYSTROKES, autocomplete=True)
            ):
                self.log.info(f"File {resources_file} does not exist!")
            self._resources_file = resources_file
        else:
            self.log.info(f"{'SGDML resource usage':-^60}\n")
            self.setupINFOS["modelpath_s0"] = question("Specify path to sGDML model for state 0: ", str, KEYSTROKES=KEYSTROKES)
            self.setupINFOS["modelpath_s1"] = question("Specify path to sGDML model for state 1: ", str, KEYSTROKES=KEYSTROKES)
            self.setupINFOS["max_memory"] = question("Specify max_memory (in MB) for sGDML: ", int, KEYSTROKES=KEYSTROKES, default=None)

        return INFOS


    def prepare(self, INFOS: dict, dir_path: str) -> None:
        create_file = link if INFOS["link_files"] else shutil.copy
        if not self._resources_file:
            with open(os.path.join(dir_path, "SGDML.resources"), "w", encoding="utf-8") as file:
                if "modelpath_s0" in self.setupINFOS:
                    file.write(f"modelpath_s0 {self.setupINFOS['modelpath_s0']}\n")
                if "modelpath_s1" in self.setupINFOS:
                    file.write(f"modelpath_s1 {self.setupINFOS['modelpath_s1']}\n")
                if "max_memory" in self.setupINFOS:
                    file.write(f"max_memory {self.setupINFOS['max_memory']}\n")
        else:
            create_file(expand_path(self._resources_file), os.path.join(dir_path, "SGDML.resources"))
        create_file(expand_path(self._template_file), os.path.join(dir_path, "SGDML.template"))

    def read_resources(self, resources_file: str = "SGDML.resources", kw_whitelist: list[str] | None = None) -> None:
        super().read_resources(resources_file, kw_whitelist)

        # Read max_memory from resources
        if "max_memory" in self.QMin.resources:
            self.QMin.resources["max_memory"] = int(self.QMin.resources["max_memory"])
        else:
            self.log.warning("max_memory not specified in SGDML.resources, using default.")
            self.QMin.resources["max_memory"] = None  # or set a default value

    def read_template(self, template_file="SGDML.template", kw_whitelist=None):
        return super().read_template(template_file, kw_whitelist)

    def setup_interface(self):
        super().setup_interface()
        max_memory = self.QMin.resources.get("max_memory", None)
        self.GDMLpredict_s0 = GDMLPredict(
            np.load(self.QMin.resources["modelpath_s0"]),
            max_memory=max_memory,
        )
        self.GDMLpredict_s1 = GDMLPredict(
            np.load(self.QMin.resources["modelpath_s1"]),
            max_memory=max_memory,
        )

    def create_restart_files(self):
        pass

    def run(self):
        pass

    def getQMout(self):
        requests = set()
        for key, val in self.QMin.requests.items():
            if val:
                requests.add(key)

        self.log.debug("Alocate space in QMout object")
        self.QMout.allocate(
            states = self.QMin.molecule["states"],
            natom = self.QMin.molecule["natom"],
            npc = self.QMin.molecule["npc"],
            requests = requests
        )
        self.log.debug("Shape of R %s", self.QMin.coords["coords"].shape)
        prediction_s0 = self.GDMLpredict_s0.predict(self.QMin.coords["coords"].reshape(1, -1))
        prediction_s1 = self.GDMLpredict_s1.predict(self.QMin.coords["coords"].reshape(1, -1))
        if self.QMin.requests["h"]:
            prediction_energy = [[prediction_s0[0], 0], [0, prediction_s1[0]]]
            self.QMout["h"] = np.asarray(prediction_energy)
            self.log.debug("Predicted energies: %s", np.asarray(prediction_energy))
            self.log.debug("Shape of predicted energies: %s", np.asarray(prediction_energy).shape)

        if self.QMin.requests["grad"]:
            prediction_grad = [-prediction_s0[1].reshape(1,self.QMin.molecule["natom"],3), -prediction_s1[1].reshape(1,self.QMin.molecule["natom"],3)]
            self.QMout["grad"] = np.asarray(prediction_grad)
            self.log.debug("Predicted gradients: %s", np.asarray(prediction_grad))
            self.log.debug("Shape of predicted gradients: %s", np.asarray(prediction_grad).shape)

        self.QMout["runtime"] = self.clock.measuretime(False)
        return self.QMout


if __name__ == "__main__":
    SHARC_SGDML().main()