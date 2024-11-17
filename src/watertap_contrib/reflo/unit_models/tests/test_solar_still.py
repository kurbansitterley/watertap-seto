#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import os
import pytest

from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.network import Port
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock, UnitModelCostingBlock, MaterialFlowBasis
from idaes.core.util.testing import initialization_tester
from idaes.core.util.exceptions import InitializationError, ConfigurationError
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
)

from watertap.core.solvers import get_solver
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)

from watertap_contrib.reflo.unit_models import SolarStill
from watertap_contrib.reflo.costing import TreatmentCosting

# Get default solver for testing
solver = get_solver()

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
test_data_path = f"{__location__}/solar_still_test_data_phoenix_az.csv"
test_data_path_too_short = (
    f"{__location__}/solar_still_test_data_phoenix_az_too_short.csv"
)

rho = 1000 * pyunits.kg / pyunits.m**3
inlet_dict = {
    "solute_list": ["TDS"],
    "mw_data": {"TDS": 31.4038218e-3},
    "material_flow_basis": MaterialFlowBasis.mass,
}

# water_yield_calc_dict = dict(
#     input_weather_file_path=test_data_path,  # path to input weather file
#     irradiance_threshold=0,  # irradiance values < threshold assumed to have negligible impact on calculation; W/m2
#     salinity=200,  # salinity of influent water; g/L
#     water_depth_basin=0.01,  # depth of water in solar still basin; m
#     length_basin=0.6,  # length of each side of basin (length=width); m
# )

water_yield_calc_dict = dict(
    input_weather_file_path="/Users/ksitterl/Documents/Python/watertap-reflo/watertap-reflo/src/watertap_contrib/reflo/unit_models/tests/solar_still_test_data_phoenix_az.csv",  # path to input weather file
    initial_salinity=200,  # initial salinity of influent water; g/L
    initial_water_depth=0.01,  # initial depth of water in solar still basin; m
    length_basin=0.6,  # length of each side of basin (length=width); m
    irradiance_threshold=20,  # irradiance values < threshold assumed to have negligible impact on calculation; W/m2
)

tds_conc = 200 * pyunits.g / pyunits.liter
daily_water_production = 100 * pyunits.m**3 / pyunits.day

flow_mass_in = pyunits.convert(
    daily_water_production * rho, to_units=pyunits.kg / pyunits.s
)
flow_mass_tds = pyunits.convert(
    daily_water_production * tds_conc, to_units=pyunits.kg / pyunits.s
)


def build_ss():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(**inlet_dict)

    m.fs.unit = SolarStill(
        property_package=m.fs.properties,
        water_yield_calculation_args=water_yield_calc_dict,
    )

    m.fs.unit.properties_in[0].flow_vol_phase[...]
    m.fs.unit.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        value(flow_mass_in)
    )
    m.fs.unit.properties_in[0].flow_mass_phase_comp["Liq", "TDS"].fix(
        value(flow_mass_tds)
    )
    m.fs.unit.properties_in[0].pressure.fix(101325)
    m.fs.unit.properties_in[0].temperature.fix(293.15)

    return m


class TestSolarStill:
    @pytest.fixture(scope="class")
    def ss_frame(self):
        m = build_ss()
        return m

    @pytest.mark.unit
    def test_config(self, ss_frame):
        m = ss_frame
        assert len(m.fs.unit.config) == 5

        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert m.fs.unit.config.property_package is m.fs.properties
        assert_units_consistent(m)

    @pytest.mark.unit
    def test_build(self, ss_frame):
        m = ss_frame

        port_list = ["inlet", "outlet", "waste"]
        for port_str in port_list:
            port = getattr(m.fs.unit, port_str)
            assert isinstance(port, Port)
            assert len(port.vars) == 3

        assert number_variables(m) == 33
        assert number_total_constraints(m) == 12
        assert number_unused_variables(m) == 18

    @pytest.mark.unit
    def test_dof(self, ss_frame):
        m = ss_frame
        assert degrees_of_freedom(m) == 0

    @pytest.mark.unit
    def test_calculate_scaling(self, ss_frame):
        m = ss_frame

        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp", 0.1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp", 10, index=("Liq", "TDS")
        )

        calculate_scaling_factors(m)

        unscaled_var_list = list(unscaled_variables_generator(m))
        assert len(unscaled_var_list) == 0

    @pytest.mark.component
    def test_initialize(self, ss_frame):
        m = ss_frame
        initialization_tester(m)

    @pytest.mark.component
    def test_solve(self, ss_frame):
        m = ss_frame
        results = solver.solve(m)
        assert_optimal_termination(results)

    @pytest.mark.unit
    def test_solution(self, ss_frame):
        m = ss_frame

        results_dict = {
            "water_yield": 1.701793,
            "length_basin": 0.6,
            "dens_mass_salt": 2.16,
            "number_stills": 163226.45,
            "total_area": 58761.522,
            "annual_water_yield": 0.621154772,
            "flow_vol_salt": 0.000107167,
            "deposition_rate": 0.01575735,
            "area_single_still": 0.36,
            "yield_per_still": 7.09e-06,
            "evaporation_rate": 1.701793,
        }
        for v, r in results_dict.items():
            ssv = getattr(m.fs.unit, v)
            if ssv.is_indexed():
                for i, s in r.items():
                    assert pytest.approx(value(ssv[i]), rel=1e-3) == s
            else:
                assert pytest.approx(value(ssv), rel=1e-3) == r

    @pytest.mark.unit
    def test_costing(self, ss_frame):
        m = ss_frame
        m.fs.costing = TreatmentCosting()
        m.fs.costing.electricity_cost.fix(0.07)
        m.fs.costing.heat_cost.fix(0.01)
        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.cost_process()
        m.fs.costing.add_LCOW(
            flow_rate=m.fs.unit.properties_out[0].flow_vol_phase["Liq"]
        )
        results = solver.solve(m)
        assert_optimal_termination(results)

        sys_cost_results = {
            "aggregate_capital_cost": 3654601.821,
            "aggregate_fixed_operating_cost": 127911.063,
            # "aggregate_variable_operating_cost": 0.0,
            "aggregate_flow_electricity": 2.625,
            # "aggregate_flow_costs": {"electricity": 0.0},
            "aggregate_direct_capital_cost": 1827300.91,
            "total_capital_cost": 3654601.821,
            "total_operating_cost": 237549.118,
            "maintenance_labor_chemical_operating_cost": 109638.054,
            "total_fixed_operating_cost": 237549.118,
            # "total_variable_operating_cost": 0.0,
            "total_annualized_cost": 646703.534,
            "LCOW": 17.705,
        }

        for v, r in sys_cost_results.items():
            sc = getattr(m.fs.costing, v)
            if sc.is_indexed():
                for i, s in r.items():
                    assert pytest.approx(value(sc[i]), rel=1e-3) == s
            else:
                assert pytest.approx(value(sc), rel=1e-3) == r

        ss_cost_results = {
            "capital_cost": 3654601.821,
            "fixed_operating_cost": 127911.063,
            "max_sw_flow": 1728.0,
            "max_fw_flow": 86.4,
            "pumping_power_required": 2.625,
            "length_piping": 481967.866,
            "capital_cost_solar_still": 2791221.587,
            "capital_cost_sw_pumps": 890.64,
            "capital_cost_fw_pumps": 178.128,
            "capital_cost_feed_tank": 1403.673,
            "capital_cost_distillate_tank": 1493.892,
            "capital_cost_underground_tank": 323.247,
            "capital_cost_excavation": 570.232,
            "capital_cost_piping": 858520.419,
            "direct_capital_cost": 1827300.91,
        }

        for v, r in ss_cost_results.items():
            ssv = getattr(m.fs.unit.costing, v)
            if ssv.is_indexed():
                for i, s in r.items():
                    assert pytest.approx(value(ssv[i]), rel=1e-3) == s
            else:
                assert pytest.approx(value(ssv), rel=1e-3) == r


@pytest.mark.unit
def test_input_data_too_short():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(**inlet_dict)

    water_yield_calc_dict["input_weather_file_path"] = test_data_path_too_short

    with pytest.raises(ValueError):
        err_msg = f"Water yield calculation for fs\\.unit requires at least "
        err_msg += f"one year of hourly weather data, but the input dataset is "
        err_msg += f"1720 hours long\\."
        m.fs.unit = SolarStill(
            property_package=m.fs.properties,
            water_yield_calculation_args=water_yield_calc_dict,
        )
