from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import logging

logging.getLogger("py4j").setLevel(logging.ERROR)

# Do not change the main function
def main():
    conf = SparkConf().setAppName("Main").setMaster("local[*]")
    sc = SparkContext(conf=conf)

    spark = SparkSession.builder.appName("2ID70").getOrCreate()

    # Execute functions equivalent to Q1, Q2, Q3, and Q4
    rdds = Q1(sc, spark)

    Q2(spark)
    Q3(rdds)
    Q4(rdds)

# Do not change the signatures (i.e., the parameters) of Q1, Q2, Q3, Q4
def Q1(sc, spark):

    patients = sc.textFile("/patients.csv")

    def parse_patient(line):
        parts = line.split(",")
        if len(parts) != 4:
            return None
        try:
            patient_id = int(parts[0])
        except ValueError:
            return None
        return patient_id, parts[1], parts[2], parts[3]

    patients = patients.map(parse_patient).filter(lambda x: x is not None)
    patients_df = spark.createDataFrame(patients, schema=["patientId", "patientName", "address", "dateOfBirth"])
    patients_df.createOrReplaceTempView("patients")

    prescriptions = sc.textFile("/prescriptions.csv")

    def parse_prescription(line):
        parts = line.split(",")
        if len(parts) != 3:
            return None
        try:
            prescription_id = int(parts[0])
            medicine_id = int(parts[1])
        except ValueError:
            return None
        return prescription_id, medicine_id, parts[2]

    prescriptions = prescriptions.map(parse_prescription).filter(lambda x: x is not None)
    prescriptions_df = spark.createDataFrame(prescriptions, schema=["prescriptionId", "medicineId", "dosage"])
    prescriptions_df.createOrReplaceTempView("prescriptions")

    diagnoses = sc.textFile("/diagnoses.csv")

    def parse_diagnosis(line):
        parts = line.split(",")
        if len(parts) != 5:
            return None
        try:
            patient_id = int(parts[0])
            doctor_id = int(parts[1])
            prescription_id = int(parts[4])
        except ValueError:
            return None
        return patient_id, doctor_id, parts[2], parts[3], prescription_id

    diagnoses = diagnoses.map(parse_diagnosis).filter(lambda x: x is not None)
    diagnoses_df = spark.createDataFrame(diagnoses, schema=["patientId", "doctorId", "date", "diagnosis", "prescriptionId"])
    diagnoses_df.createOrReplaceTempView("diagnoses")

    # Create views in Spark session for SparkSQL (if needed)

    return patients, prescriptions, diagnoses

# Do not change the signatures (i.e., the parameters) of Q1, Q2, Q3, Q4
def Q2(spark):

    q21 = spark.sql("""SELECT COUNT(*) AS num_patients FROM patients 
                    WHERE dateOfBirth LIKE '1999-%'""")
    num_patients = q21.collect()[0]['num_patients']
    print(f">> [q21: {num_patients}]")

    q22 = spark.sql("""SELECT date, COUNT(*) AS diag_count 
                    FROM diagnoses 
                    WHERE date LIKE '2024-%' 
                    GROUP BY date
                    ORDER BY diag_count DESC 
                    LIMIT 1""")
    date = q22.collect()[0]['date']
    print(f">> [q22: {date}]")

    q23 = spark.sql("""
            SELECT d.date, pc.medicine_count
            FROM diagnoses d
            JOIN (
                SELECT prescriptionId, COUNT(*) AS medicine_count
                FROM prescriptions
                GROUP BY prescriptionId
            ) pc
            ON d.prescriptionId = pc.prescriptionId
            WHERE d.date LIKE '2024-%'
            ORDER BY pc.medicine_count DESC
            LIMIT 1
        """)
    date = q23.collect()[0]['date']
    print(f">> [q23: {date}]")

# Do not change the signatures (i.e., the parameters) of Q1, Q2, Q3, Q4
def Q3(rdds):
    
    patients, prescriptions, diagnoses = rdds

    q31 = patients.filter(lambda x: x[3].startswith("1999")).count()
    print(f">> [q31: {q31}]")

    diagnoses_2024 = diagnoses.filter(lambda x: x[2].startswith("2024"))
    date_counts = diagnoses_2024.map(lambda x: (x[2], 1)).reduceByKey(lambda a, b: a + b)
    q32 = date_counts.max(lambda x: x[1])[0]
    print(f">> [q32: {q32}]")

    medicine_counts = prescriptions.map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
    # Output: (prescriptionId, medicine_count)

    # Step 2: Join with diagnoses on prescriptionId
    diagnoses_with_meds = diagnoses.map(lambda x: (x[4], x[2])).join(medicine_counts)  # (prescriptionId, (date, medicine_count))

    # Step 3: Filter for 2024 diagnoses
    diagnoses_2024 = diagnoses_with_meds.filter(lambda x: x[1][0].startswith("2024"))

    q33 = diagnoses_2024.map(lambda x: (x[1][0], x[1][1])).reduce(lambda a, b: a if a[1] > b[1] else b)[0]
    print(f">> [q33: {q33}]")

# Do not change the signatures (i.e., the parameters) of Q1, Q2, Q3, Q4
def Q4(rdds):

    _, _, diagnoses = rdds
    
    q4 = 0

    print(f">> [q4: {q4}]")

if __name__ == "__main__":
    main()
