from dataclasses import dataclass


@dataclass
class Query:
    STORAGE_SOC_QUERY = """
            SELECT
                keyvalue,
                month,
                day,
                hour,
                value,
                interval,
                units
            FROM
                `ReportVariableWithTime`
            WHERE name = 'Electric Storage Simple Charge State'
            """
    TYPICAL_SUMMER_DAY_SOC_QUERY = """
                SELECT
            month,
                day,
                hour,
                sum(case when name = 'Generator Produced DC Electricity Rate' then value else 0 end) as pv_value,
                sum(case when name = 'Electric Storage Charge Power' then value else 0 end) as storage_charge_value,
                sum(case when name = 'Electric Storage Discharge Power' then value else 0 end) as storage_discharge_value,
                sum(case when name = 'Facility Total Electricity Demand Rate' then value else 0 end) as demand_value
            FROM
                `ReportVariableWithTime`
            WHERE
                name in ('Generator Produced DC Electricity Rate', 'Electric Storage Charge Power', 'Electric Storage Discharge Power', 'Facility Total Electricity Demand Rate') and month = 7 and day in (12,13,14,15,16,17,18)
            GROUP BY
                month, day, hour
    """

    TYPICAL_WINTER_DAY_SOC_QUERY = """
                SELECT
            month,
                day,
                hour,
                sum(case when name = 'Generator Produced DC Electricity Rate' then value else 0 end) as pv_value,
                sum(case when name = 'Electric Storage Charge Power' then value else 0 end) as storage_charge_value,
                sum(case when name = 'Electric Storage Discharge Power' then value else 0 end) as storage_discharge_value,
                sum(case when name = 'Facility Total Electricity Demand Rate' then value else 0 end) as demand_value
            FROM
                `ReportVariableWithTime`
            WHERE
                name in ('Generator Produced DC Electricity Rate', 'Electric Storage Charge Power', 'Electric Storage Discharge Power', 'Facility Total Electricity Demand Rate') and month = 1 and day in (12,13,14,15,16,17,18)
            GROUP BY
                month, day, hour
    """
