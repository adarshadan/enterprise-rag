from locust import HttpUser, task, between
import random


class RAGUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """
        Runs once per virtual user.
        First user loads the document.
        Subsequent users will receive
        'already_loaded' response.
        """

        payload = {
            "file_path": "./data/indian-penal-code.pdf",
            "collection": "test"
        }

        try:
            response = self.client.post(
                "/load",
                json=payload,
                name="/load"
            )

            print(
                f"LOAD STATUS: {response.status_code} | "
                f"{response.text}"
            )

        except Exception as e:
            print(f"LOAD FAILED: {e}")

        # Verify system health
        try:
            response = self.client.get(
                "/health",
                name="/health"
            )

            print(
                f"HEALTH STATUS: {response.status_code} | "
                f"{response.text}"
            )

        except Exception as e:
            print(f"HEALTH CHECK FAILED: {e}")

    @task(5)
    def section_query(self):

        queries = [
            "section 420",
            "section 302",
            "section 378",
            "section 120B",
            "section 34"
        ]

        query = random.choice(queries)

        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True,
            name="/query-section"
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Status={response.status_code}, "
                    f"Response={response.text}"
                )

    @task(3)
    def legal_definition_query(self):

        queries = [
            "what is murder",
            "what is theft",
            "what is cheating",
            "what is criminal conspiracy",
            "what is kidnapping"
        ]

        query = random.choice(queries)

        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True,
            name="/query-definition"
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Status={response.status_code}, "
                    f"Response={response.text}"
                )

    @task(2)
    def punishment_query(self):

        queries = [
            "punishment for theft",
            "punishment for murder",
            "punishment for cheating",
            "punishment for kidnapping"
        ]

        query = random.choice(queries)

        with self.client.post(
            "/query",
            json={"query": query},
            catch_response=True,
            name="/query-punishment"
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Status={response.status_code}, "
                    f"Response={response.text}"
                )