from pprint import pprint

from pyomo.environ import (
    ConcreteModel,
    value,
    Var,
    assert_optimal_termination,
    units as pyunits,
)
from pyomo.util.check_units import assert_units_consistent
from pyomo.util.calc_var_value import calculate_variable_from_constraint as cvc


from idaes.core.util.scaling import *
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core import FlowsheetBlock, UnitModelCostingBlock

from watertap.core.solvers import get_solver
from watertap.core.util.model_diagnostics.infeasible import *
from watertap.property_models.unit_specific.cryst_prop_pack import (
    NaClParameterBlock,
)
from watertap.property_models.water_prop_pack import WaterParameterBlock

from watertap_contrib.reflo.costing import TreatmentCosting
from watertap_contrib.reflo.unit_models.multi_effect_crystallizer import (
    MultiEffectCrystallizer,
)
from watertap_contrib.reflo.unit_models.crystallizer_effect import CrystallizerEffect


solver = get_solver()
rho = 1000 * pyunits.kg / pyunits.m**3
feed_pressure = 101325
feed_temperature = 273.15 + 20


def build_kbhdp_mec(
    flow_in=4,  # MGD
    kbhdp_salinity=12,  # g/L
    assumed_lssro_recovery=0.9,
    number_effects=4,
    crystallizer_yield=0.5,
    saturated_steam_pressure_gage=3,
    heat_transfer_coefficient=0.1,
    eps=1e-8,
):
    """
    Build MultiEffectCrystallizer for KBHDP case study.
        flow_in: volumetric flow rate in MGD
        kbhdp_salinity: salinity of KBHDP brine
        assumed_lssro_recovery: recovery of LSRRO process that is assumed to be before MEC; used to approximate influent concentration
        number_effects: number of effects for MEC model
    """

    global flow_mass_phase_water_total, flow_mass_phase_salt_total, conc_in

    atm_pressure = 101325 * pyunits.Pa

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = NaClParameterBlock()
    m.fs.vapor_properties = WaterParameterBlock()

    m.fs.mec = mec = MultiEffectCrystallizer(
        property_package=m.fs.properties,
        property_package_vapor=m.fs.vapor_properties,
        number_effects=number_effects,
    )

    operating_pressures = [0.45, 0.25, 0.208, 0.095]

    conc_in = pyunits.convert(
        (kbhdp_salinity * pyunits.g / pyunits.liter) / (1 - assumed_lssro_recovery),
        to_units=pyunits.kg / pyunits.m**3,
    )

    ### TOTAL GOING INTO MEC
    flow_vol_in = pyunits.convert(
        flow_in * pyunits.Mgallons / pyunits.day, to_units=pyunits.m**3 / pyunits.s
    )
    flow_mass_phase_water_total = pyunits.convert(
        flow_vol_in * rho, to_units=pyunits.kg / pyunits.s
    )
    flow_mass_phase_salt_total = pyunits.convert(
        flow_vol_in * conc_in, to_units=pyunits.kg / pyunits.s
    )

    ### TOTAL INTO EACH EFFECT INITIAL

    flow_mass_phase_water_per = flow_mass_phase_water_total / number_effects
    flow_mass_phase_salt_per = flow_mass_phase_salt_total / number_effects

    saturated_steam_pressure = atm_pressure + pyunits.convert(
        saturated_steam_pressure_gage * pyunits.bar, to_units=pyunits.Pa
    )

    for (_, eff), op_pressure in zip(mec.effects.items(), operating_pressures):

        eff.effect.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].fix(
            flow_mass_phase_water_per
        )
        eff.effect.properties_in[0].flow_mass_phase_comp["Liq", "NaCl"].fix(
            flow_mass_phase_salt_per
        )

        eff.effect.properties_in[0].pressure.fix(feed_pressure)
        eff.effect.properties_in[0].temperature.fix(feed_temperature)

        eff.effect.properties_in[0].flow_mass_phase_comp["Sol", "NaCl"].fix(eps)
        eff.effect.properties_in[0].flow_mass_phase_comp["Vap", "H2O"].fix(eps)
        eff.effect.properties_in[0].conc_mass_phase_comp[...]

        eff.effect.crystallization_yield["NaCl"].fix(crystallizer_yield)
        eff.effect.crystal_growth_rate.fix()
        eff.effect.souders_brown_constant.fix()
        eff.effect.crystal_median_length.fix()

        eff.effect.pressure_operating.fix(
            pyunits.convert(op_pressure * pyunits.bar, to_units=pyunits.Pa)
        )
        eff.effect.overall_heat_transfer_coefficient.set_value(
            heat_transfer_coefficient
        )

    first_effect = m.fs.mec.effects[1].effect

    first_effect.overall_heat_transfer_coefficient.fix(heat_transfer_coefficient)
    first_effect.heating_steam[0].pressure_sat
    first_effect.heating_steam[0].dh_vap_mass
    first_effect.heating_steam[0].flow_mass_phase_comp["Liq", "H2O"].unfix()
    first_effect.heating_steam[0].flow_mass_phase_comp["Vap", "H2O"].unfix()
    first_effect.heating_steam.calculate_state(
        var_args={
            ("flow_mass_phase_comp", ("Liq", "H2O")): 0,
            ("pressure", None): saturated_steam_pressure,
            ("pressure_sat", None): saturated_steam_pressure,
        },
        hold_state=True,
    )
    first_effect.heating_steam[0].flow_mass_phase_comp["Vap", "H2O"].unfix()

    m.fs.mec.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].fix(
        flow_mass_phase_water_total
    )
    m.fs.mec.control_volume.properties_in[0].flow_mass_phase_comp["Liq", "NaCl"].fix(
        flow_mass_phase_salt_total
    )

    m.fs.mec.control_volume.properties_in[0].flow_mass_phase_comp["Sol", "NaCl"].fix(0)
    m.fs.mec.control_volume.properties_in[0].pressure.fix(feed_pressure)
    m.fs.mec.control_volume.properties_in[0].temperature.fix(feed_temperature)

    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1 / value(flow_mass_phase_water_per),
        index=("Liq", "H2O"),
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp",
        1 / value(flow_mass_phase_salt_per),
        index=("Liq", "NaCl"),
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 10, index=("Vap", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-2, index=("Sol", "NaCl")
    )
    m.fs.vapor_properties.set_default_scaling(
        "flow_mass_phase_comp", 1e-2, index=("Vap", "H2O")
    )
    m.fs.vapor_properties.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    return m


if __name__ == "__main__":

    m = build_kbhdp_mec(
        # m=m
        # flow_in=4,  # MGD
        # kbhdp_salinity=12,  # g/L
        # assumed_lssro_recovery=0.95,
        # number_effects=4,
        crystallizer_yield=0.5,
        # eps=1e-12,
        # saturated_steam_pressure_gage=3,
        # heat_transfer_coefficient=0.1
    )

    mec = m.fs.mec
    fe = mec.effects[1].effect

    print(f"dof = {degrees_of_freedom(m)}")

    calculate_scaling_factors(m)

    try:
        mec.initialize()
    except:
        print_infeasible_constraints(mec)
        print_variables_close_to_bounds(mec)

    print(f"dof = {degrees_of_freedom(m)}")

    results = solver.solve(m)
    assert_optimal_termination(results)
